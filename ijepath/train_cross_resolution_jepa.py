import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import tqdm

from ijepath.datasets.cross_resolution_loader_factory import (
    make_cross_resolution_loader,
)
from ijepath.datasets.cross_resolution_wsi_dataset import snap_size_to_patch_multiple
from ijepath.config_logging import log_and_write_config
from ijepath.helper import init_model, init_opt, load_checkpoint
from ijepath.log.tracker import (
    finish_wandb,
    initialize_wandb,
    log_epoch_dict,
    save_run_config_to_wandb,
    update_log_dict,
)
from ijepath.utils.distributed import AllReduce, init_distributed
from ijepath.utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger
from ijepath.utils.log_utils import setup_logging
from ijepath.utils.tensors import repeat_interleave_batch

default_checkpoint_every_epochs = 50
default_step_log_every_iters = 0

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = logging.getLogger("ijepath")


def pool_predictor_tokens(z_tokens: torch.Tensor) -> torch.Tensor:
    """Pool predictor tokens into one embedding per predicted target footprint."""
    return z_tokens.mean(dim=1)


def flatten_teacher_targets_for_predictor_order(
    teacher: torch.Tensor,
    batch_size: int,
    num_pred_masks: int,
    num_enc_masks: int,
) -> torch.Tensor:
    """Match teacher ordering with predictor output ordering.

    `teacher` shape is [B, K, D] where K=num_pred_masks.
    Predictor output order is pred-mask-major, and repeated per encoder mask.
    """
    if teacher.shape[0] != batch_size:
        raise ValueError("teacher batch dimension mismatch")
    if teacher.shape[1] != num_pred_masks:
        raise ValueError("teacher target dimension mismatch")

    teacher = teacher.permute(1, 0, 2).reshape(num_pred_masks * batch_size, -1)
    if num_enc_masks > 1:
        teacher = repeat_interleave_batch(teacher, batch_size, repeat=num_enc_masks)
    return teacher


def should_log_iteration(itr: int, step_log_every_iters: int, loss: float) -> bool:
    if np.isnan(loss) or np.isinf(loss):
        return True
    if step_log_every_iters <= 0:
        return False
    return (itr % step_log_every_iters) == 0


def resolve_checkpoint_every_epochs(logging_cfg: dict) -> int:
    checkpoint_every_epochs = int(
        logging_cfg.get("checkpoint_every_epochs", default_checkpoint_every_epochs)
    )
    if checkpoint_every_epochs <= 0:
        raise ValueError("logging.checkpoint_every_epochs must be > 0")
    return checkpoint_every_epochs


def resolve_use_bfloat16(requested_use_bfloat16: bool, cuda_available: bool) -> bool:
    return bool(requested_use_bfloat16 and cuda_available)


def build_epoch_train_results(
    *,
    loss_avg: float,
    loss_min: float,
    loss_max: float,
    mask_a: float,
    mask_b: float,
    iter_time_ms: float,
    epoch_time_s: float,
    images_seen: int,
    images_per_sec: float,
    iterations_per_sec: float,
    lr: float,
    wd: float,
    global_batch_size: int,
) -> dict[str, float | int]:
    return {
        "loss": float(loss_avg),
        "loss_min": float(loss_min),
        "loss_max": float(loss_max),
        "mask_a": float(mask_a),
        "mask_b": float(mask_b),
        "iter_time_ms": float(iter_time_ms),
        "epoch_time_s": float(epoch_time_s),
        "images_seen": int(images_seen),
        "images_per_sec": float(images_per_sec),
        "iterations_per_sec": float(iterations_per_sec),
        "lr": float(lr),
        "wd": float(wd),
        "global_batch_size": int(global_batch_size),
    }


def get_train_step_csv_columns() -> tuple[tuple[str, str], ...]:
    return (
        ("%d", "epoch"),
        ("%d", "iteration"),
        ("%.5f", "loss"),
        ("%.5f", "context_keep_tokens"),
        ("%.5f", "target_predict_tokens"),
        ("%.3f", "iteration_time_ms"),
        ("%.8f", "learning_rate"),
        ("%.8f", "weight_decay"),
    )


def build_step_log_line(
    *,
    epoch: int,
    iteration: int,
    loss_avg: float,
    context_keep_tokens: float,
    target_predict_tokens: float,
    weight_decay: float,
    learning_rate: float,
    max_memory_mb: float,
    iteration_time_ms: float,
) -> str:
    return (
        f"epoch={int(epoch)} iteration={int(iteration)} "
        f"loss_avg={float(loss_avg):.3f} "
        f"context_keep_tokens={float(context_keep_tokens):.1f} "
        f"target_predict_tokens={float(target_predict_tokens):.1f} "
        f"weight_decay={float(weight_decay):.2e} "
        f"learning_rate={float(learning_rate):.2e} "
        f"max_memory_mb={float(max_memory_mb):.2e} "
        f"iteration_time_ms={float(iteration_time_ms):.1f}"
    )


def build_grad_stats_log_line(
    *,
    epoch: int,
    iteration: int,
    first_layer_grad_norm: float,
    last_layer_grad_norm: float,
    grad_min: float,
    grad_max: float,
) -> str:
    return (
        f"epoch={int(epoch)} iteration={int(iteration)} "
        f"first_layer_grad_norm={float(first_layer_grad_norm):.2e} "
        f"last_layer_grad_norm={float(last_layer_grad_norm):.2e} "
        f"grad_norm_min={float(grad_min):.2e} "
        f"grad_norm_max={float(grad_max):.2e}"
    )


def copy_matching_state_dict_params(source: torch.nn.Module, target: torch.nn.Module) -> tuple[int, int]:
    source_state = source.state_dict()
    target_state = target.state_dict()

    matched = 0
    for name, target_value in target_state.items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_state[name] = source_value.detach().clone()
        matched += 1

    target.load_state_dict(target_state)
    skipped = int(len(target_state) - matched)
    return matched, skipped


def main(
    args,
    resume_preempt: bool = False,
    distributed_state: tuple[int, int] | None = None,
):
    # -- META
    requested_use_bfloat16 = bool(args["meta"]["use_bfloat16"])
    use_bfloat16 = resolve_use_bfloat16(
        requested_use_bfloat16=requested_use_bfloat16,
        cuda_available=torch.cuda.is_available(),
    )
    architecture = args["meta"]["architecture"]
    patch_size = int(args["meta"]["patch_size"])
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if requested_use_bfloat16 and not use_bfloat16:
        logger.warning("Disabled bfloat16 AMP because CUDA is unavailable in this runtime.")

    # -- DATA
    batch_size_per_gpu = int(args["data"].get("batch_size_per_gpu", args["data"].get("batch_size")))
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]

    anchor_catalog_csv = args["data"]["anchor_catalog_csv"]
    context_mpp = float(args["data"]["context_mpp"])
    target_mpp = float(args["data"]["target_mpp"])
    context_fov_um = float(args["data"]["context_fov_um"])
    target_fov_um = float(args["data"]["target_fov_um"])
    targets_per_context = int(args["data"]["targets_per_context"])
    seed = int(args["data"]["seed"])
    samples_per_epoch = args["data"].get("samples_per_epoch", None)
    if samples_per_epoch is not None:
        samples_per_epoch = int(samples_per_epoch)
    spacing_tolerance = float(args["data"].get("spacing_tolerance", 0.05))
    min_target_tissue_fraction = float(
        args["data"].get("min_target_tissue_fraction", args["data"].get("min_tissue_fraction", 0.25))
    )
    insufficient_target_policy = str(args["data"].get("insufficient_target_policy", "skip_anchor"))
    min_target_tissue_fraction_floor = args["data"].get("min_target_tissue_fraction_floor", None)
    if min_target_tissue_fraction_floor is not None:
        min_target_tissue_fraction_floor = float(min_target_tissue_fraction_floor)
    min_target_tissue_fraction_step = float(args["data"].get("min_target_tissue_fraction_step", 0.05))
    wsi_backend = str(args["data"].get("wsi_backend", "openslide"))

    # -- MASK
    num_enc_masks = args["mask"]["num_enc_masks"]
    min_keep = args["mask"]["min_keep"]
    context_size_raw_px = max(1, int(round(context_fov_um / context_mpp)))
    target_size_raw_px = max(1, int(round(target_fov_um / target_mpp)))
    context_input_size_px = snap_size_to_patch_multiple(
        size_px=context_size_raw_px,
        patch_size=patch_size,
    )
    target_input_size_px = snap_size_to_patch_multiple(
        size_px=target_size_raw_px,
        patch_size=patch_size,
    )
    if context_input_size_px != context_size_raw_px:
        logger.info(
            f"Snapped context size to patch multiple: raw={context_size_raw_px} "
            f"snapped={context_input_size_px} patch={patch_size}"
        )
    if target_input_size_px != target_size_raw_px:
        logger.info(
            f"Snapped target size to patch multiple: raw={target_size_raw_px} "
            f"snapped={target_input_size_px} patch={patch_size}"
        )

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    step_log_every_iters = int(
        args["logging"].get("step_log_every_iters", default_step_log_every_iters)
    )
    if step_log_every_iters < 0:
        raise ValueError("logging.step_log_every_iters must be >= 0")
    checkpoint_every_epochs = resolve_checkpoint_every_epochs(args["logging"])

    os.makedirs(folder, exist_ok=True)

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if distributed_state is None:
        world_size, rank = init_distributed()
    else:
        world_size, rank = int(distributed_state[0]), int(distributed_state[1])
    logger_level = logging.INFO if rank == 0 else logging.ERROR
    setup_logging(output=folder, level=logger_level)

    global_batch_size = batch_size_per_gpu * world_size
    args.setdefault("data", {})
    args["data"]["batch_size_per_gpu"] = batch_size_per_gpu
    args["data"]["global_batch_size"] = global_batch_size
    args["data"]["world_size"] = world_size
    args["data"]["context_input_size_px"] = context_input_size_px
    args["data"]["target_input_size_px"] = target_input_size_px

    wandb_cfg = dict(args.get("wandb", {}) or {})
    wandb_enabled = rank == 0 and bool(wandb_cfg.get("enable", False))

    if rank == 0:
        cfg_path = log_and_write_config(args, output_dir=folder, logger=logger)
        logger.info(f"Wrote resolved run config to {cfg_path}")
        if wandb_enabled:
            wandb_run = initialize_wandb(args, key=os.environ.get("WANDB_API_KEY"))
            save_run_config_to_wandb(wandb_run, args)
            wandb_run.save(cfg_path)

    logger.info(
        f"Distributed context ready: rank={rank} world_size={world_size} "
        f"is_main_process={rank == 0}"
    )
    logger.info(
        f"Batch sizing: batch_size_per_gpu={batch_size_per_gpu} "
        f"global_batch_size={global_batch_size}"
    )
    logger.info(
        "Iteration logging cadence: "
        f"step_log_every_iters={step_log_every_iters} (0 disables per-step logs)"
    )
    logger.info(f"Checkpoint cadence: checkpoint_every_epochs={checkpoint_every_epochs}")

    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = os.path.join(folder, r_file) if (load_model and r_file is not None) else latest_path

    csv_logger = CSVLogger(log_file, *get_train_step_csv_columns())

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=context_input_size_px,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        architecture=architecture,
    )
    target_encoder, _unused_target_predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=target_input_size_px,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        architecture=architecture,
    )
    del _unused_target_predictor
    matched_params, skipped_params = copy_matching_state_dict_params(
        source=encoder,
        target=target_encoder,
    )
    logger.info(
        f"Initialized target encoder from context encoder: "
        f"matched_state_entries={matched_params} skipped_state_entries={skipped_params}"
    )

    unsupervised_dataset, unsupervised_loader, unsupervised_sampler = make_cross_resolution_loader(
        batch_size=batch_size_per_gpu,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        anchor_catalog_csv=anchor_catalog_csv,
        patch_size=patch_size,
        context_mpp=context_mpp,
        target_mpp=target_mpp,
        context_fov_um=context_fov_um,
        target_fov_um=target_fov_um,
        targets_per_context=targets_per_context,
        seed=seed,
        spacing_tolerance=spacing_tolerance,
        min_target_tissue_fraction=min_target_tissue_fraction,
        insufficient_target_policy=insufficient_target_policy,
        min_target_tissue_fraction_floor=min_target_tissue_fraction_floor,
        min_target_tissue_fraction_step=min_target_tissue_fraction_step,
        min_keep=min_keep,
        num_enc_masks=num_enc_masks,
        backend=wsi_backend,
        samples_per_epoch=samples_per_epoch,
    )
    ipe = len(unsupervised_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1
    if use_ddp:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    target_named_params = dict(target_encoder.named_parameters())
    ema_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
    for name_q, param_q in encoder.named_parameters():
        param_k = target_named_params.get(name_q)
        if param_k is None or param_q.shape != param_k.shape:
            continue
        ema_pairs.append((param_q, param_k))
    logger.info(
        f"EMA parameter pairing ready: matched_pairs={len(ema_pairs)} "
        f"skipped_pairs={len(target_named_params) - len(ema_pairs)}"
    )

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size_per_gpu": batch_size_per_gpu,
            "global_batch_size": global_batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_every_epochs == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))

    try:
        with tqdm.tqdm(
            range(start_epoch, num_epochs),
            desc="I-JEPATH Pretraining",
            unit=" epoch",
            ncols=100,
            leave=True,
            initial=start_epoch,
            total=num_epochs,
            disable=rank != 0,
        ) as epoch_bar:
            for epoch in epoch_bar:
                epoch_start = time.time()
                logger.info(f"Epoch {epoch + 1}")
                if hasattr(unsupervised_dataset, "set_epoch"):
                    unsupervised_dataset.set_epoch(epoch)
                unsupervised_sampler.set_epoch(epoch)

                loss_meter = AverageMeter()
                maskA_meter = AverageMeter()
                maskB_meter = AverageMeter()
                time_meter = AverageMeter()
                epoch_last_lr = float("nan")
                epoch_last_wd = float("nan")

                if wandb_enabled:
                    epoch_log: dict[str, float | int] = {"epoch": epoch + 1}
                else:
                    epoch_log = {}

                images_per_iter = int(global_batch_size)
                epoch_images_target = int(ipe * images_per_iter)
                with tqdm.tqdm(
                    total=epoch_images_target,
                    desc=f"Epoch [{epoch + 1}/{num_epochs}]",
                    unit=" img",
                    unit_scale=True,
                    ncols=100,
                    leave=False,
                    disable=rank != 0,
                ) as iter_bar:
                    for itr, (batch_data, masks_enc, masks_pred) in enumerate(unsupervised_loader):
                        context_imgs = batch_data["context_images"].to(device, non_blocking=True)
                        target_imgs = batch_data["target_images"].to(device, non_blocking=True)
                        masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
                        masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

                        maskA_meter.update(len(masks_enc[0][0]))
                        maskB_meter.update(len(masks_pred[0][0]))

                        def train_step():
                            _new_lr = scheduler.step()
                            _new_wd = wd_scheduler.step()

                            def forward_target():
                                with torch.no_grad():
                                    bsz, ntargets, channels, height, width = target_imgs.shape
                                    targets_flat = target_imgs.view(bsz * ntargets, channels, height, width)
                                    h = target_encoder(targets_flat)
                                    h = F.layer_norm(h, (h.size(-1),))
                                    h = h.mean(dim=1).view(bsz, ntargets, -1)
                                    h = flatten_teacher_targets_for_predictor_order(
                                        teacher=h,
                                        batch_size=bsz,
                                        num_pred_masks=len(masks_pred),
                                        num_enc_masks=len(masks_enc),
                                    )
                                    return h

                            def forward_context():
                                z = encoder(context_imgs, masks_enc)
                                z = predictor(z, masks_enc, masks_pred)
                                z = pool_predictor_tokens(z)
                                return z

                            def loss_fn(z, h):
                                loss = F.smooth_l1_loss(z, h)
                                loss = AllReduce.apply(loss)
                                return loss

                            autocast_context = (
                                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                                if use_bfloat16
                                else nullcontext()
                            )
                            with autocast_context:
                                h = forward_target()
                                z = forward_context()
                                loss = loss_fn(z, h)

                            if use_bfloat16 and scaler is not None:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                            grad_stats = grad_logger(encoder.named_parameters())
                            optimizer.zero_grad()

                            with torch.no_grad():
                                m = next(momentum_scheduler)
                                for param_q, param_k in ema_pairs:
                                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                            return (float(loss), _new_lr, _new_wd, grad_stats)

                        (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
                        epoch_last_lr = float(_new_lr)
                        epoch_last_wd = float(_new_wd)

                        loss_meter.update(loss)
                        time_meter.update(etime)

                        csv_logger.log(
                            epoch + 1,
                            itr,
                            loss,
                            maskA_meter.val,
                            maskB_meter.val,
                            etime,
                            epoch_last_lr,
                            epoch_last_wd,
                        )
                        if rank == 0:
                            iter_bar.update(images_per_iter)
                            iter_bar.set_postfix(
                                loss=f"{loss_meter.avg:.3f}",
                                lr=f"{epoch_last_lr:.2e}",
                                wd=f"{epoch_last_wd:.2e}",
                            )

                        if should_log_iteration(
                            itr=itr,
                            step_log_every_iters=step_log_every_iters,
                            loss=float(loss),
                        ):
                            if torch.cuda.is_available():
                                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0**2
                            else:
                                max_mem_mb = 0.0
                            logger.info(
                                build_step_log_line(
                                    epoch=epoch + 1,
                                    iteration=itr,
                                    loss_avg=float(loss_meter.avg),
                                    context_keep_tokens=float(maskA_meter.avg),
                                    target_predict_tokens=float(maskB_meter.avg),
                                    weight_decay=float(epoch_last_wd),
                                    learning_rate=float(epoch_last_lr),
                                    max_memory_mb=float(max_mem_mb),
                                    iteration_time_ms=float(time_meter.avg),
                                )
                            )
                            if grad_stats is not None:
                                logger.info(
                                    build_grad_stats_log_line(
                                        epoch=epoch + 1,
                                        iteration=itr,
                                        first_layer_grad_norm=float(grad_stats.first_layer),
                                        last_layer_grad_norm=float(grad_stats.last_layer),
                                        grad_min=float(grad_stats.min),
                                        grad_max=float(grad_stats.max),
                                    )
                                )

                        assert not np.isnan(loss), "loss is nan"

                epoch_seconds = time.time() - epoch_start
                epoch_images_seen = int(ipe * global_batch_size)
                epoch_images_per_sec = (
                    float(epoch_images_seen) / float(epoch_seconds)
                    if epoch_seconds > 0
                    else 0.0
                )
                epoch_iterations_per_sec = (
                    float(ipe) / float(epoch_seconds)
                    if epoch_seconds > 0
                    else 0.0
                )
                logger.info(
                    f"Epoch {epoch + 1} summary: "
                    f"loss(avg/min/max)={float(loss_meter.avg):.3f}/{float(loss_meter.min):.3f}/{float(loss_meter.max):.3f} "
                    f"lr={float(epoch_last_lr):.2e} wd={float(epoch_last_wd):.2e} "
                    f"images={epoch_images_seen} images_per_sec={epoch_images_per_sec:.1f} "
                    f"iterations_per_sec={epoch_iterations_per_sec:.2f} iter_time_ms={float(time_meter.avg):.1f} "
                    f"masks(avg)={float(maskA_meter.avg):.1f}/{float(maskB_meter.avg):.1f}"
                )
                save_checkpoint(epoch + 1)

                if wandb_enabled:
                    train_results = build_epoch_train_results(
                        loss_avg=float(loss_meter.avg),
                        loss_min=float(loss_meter.min),
                        loss_max=float(loss_meter.max),
                        mask_a=float(maskA_meter.avg),
                        mask_b=float(maskB_meter.avg),
                        iter_time_ms=float(time_meter.avg),
                        epoch_time_s=float(epoch_seconds),
                        images_seen=int(epoch_images_seen),
                        images_per_sec=float(epoch_images_per_sec),
                        iterations_per_sec=float(epoch_iterations_per_sec),
                        lr=float(epoch_last_lr),
                        wd=float(epoch_last_wd),
                        global_batch_size=int(global_batch_size),
                    )
                    update_log_dict("train", train_results, epoch_log, step="epoch")
                    log_epoch_dict(epoch_log, epoch=epoch + 1)
    finally:
        if wandb_enabled:
            finish_wandb()


if __name__ == "__main__":
    raise SystemExit("Use ijepath.train.main entrypoint")
