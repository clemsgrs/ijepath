import copy
import logging
import os
import time

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

log_freq = 10
checkpoint_freq = 50

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


def main(args, resume_preempt: bool = False):
    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- DATA
    batch_size_per_gpu = int(args["data"].get("batch_size_per_gpu", args["data"].get("batch_size")))
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    crop_size = args["data"]["crop_size"]

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
    patch_size = args["mask"]["patch_size"]
    num_enc_masks = args["mask"]["num_enc_masks"]
    min_keep = args["mask"]["min_keep"]

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

    os.makedirs(folder, exist_ok=True)

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger_level = logging.INFO if rank == 0 else logging.ERROR
    setup_logging(output=folder, level=logger_level)

    global_batch_size = batch_size_per_gpu * world_size
    args.setdefault("data", {})
    args["data"]["batch_size_per_gpu"] = batch_size_per_gpu
    args["data"]["global_batch_size"] = global_batch_size
    args["data"]["world_size"] = world_size

    wandb_cfg = dict(args.get("wandb", {}) or {})
    wandb_enabled = rank == 0 and bool(wandb_cfg.get("enable", False))

    if rank == 0:
        cfg_path = log_and_write_config(args, output_dir=folder, logger=logger)
        if wandb_enabled:
            wandb_run = initialize_wandb(args, key=os.environ.get("WANDB_API_KEY"))
            save_run_config_to_wandb(wandb_run, args)
            wandb_run.save(cfg_path)

    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    logger.info(
        "Batch sizing: per_gpu=%d global=%d",
        batch_size_per_gpu,
        global_batch_size,
    )

    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = os.path.join(folder, r_file) if (load_model and r_file is not None) else latest_path

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "mask-A"),
        ("%.5f", "mask-B"),
        ("%d", "time (ms)"),
    )

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    unsupervised_dataset, unsupervised_loader, unsupervised_sampler = make_cross_resolution_loader(
        batch_size=batch_size_per_gpu,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        anchor_catalog_csv=anchor_catalog_csv,
        crop_size=crop_size,
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
            if (epoch + 1) % checkpoint_freq == 0:
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
                logger.info("Epoch %d", epoch + 1)
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

                with tqdm.tqdm(
                    unsupervised_loader,
                    desc=f"Epoch [{epoch + 1}/{num_epochs}]",
                    unit=" it",
                    ncols=100,
                    leave=False,
                    total=ipe,
                    disable=rank != 0,
                ) as iter_bar:
                    for itr, (batch_data, masks_enc, masks_pred) in enumerate(iter_bar):
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

                            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                                h = forward_target()
                                z = forward_context()
                                loss = loss_fn(z, h)

                            if use_bfloat16:
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
                                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                            return (float(loss), _new_lr, _new_wd, grad_stats)

                        (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
                        epoch_last_lr = float(_new_lr)
                        epoch_last_wd = float(_new_wd)

                        loss_meter.update(loss)
                        time_meter.update(etime)

                        csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                        if rank == 0:
                            iter_bar.set_postfix(
                                loss=f"{loss_meter.avg:.3f}",
                                lr=f"{epoch_last_lr:.2e}",
                                wd=f"{epoch_last_wd:.2e}",
                            )

                        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                            if torch.cuda.is_available():
                                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0**2
                            else:
                                max_mem_mb = 0.0
                            logger.info(
                                "[%d, %5d] loss: %.3f masks: %.1f %.1f [wd: %.2e] [lr: %.2e] [mem: %.2e] (%.1f ms)"
                                % (
                                    epoch + 1,
                                    itr,
                                    loss_meter.avg,
                                    maskA_meter.avg,
                                    maskB_meter.avg,
                                    epoch_last_wd,
                                    epoch_last_lr,
                                    max_mem_mb,
                                    time_meter.avg,
                                )
                            )
                            if grad_stats is not None:
                                logger.info(
                                    "[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                                    % (
                                        epoch + 1,
                                        itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max,
                                    )
                                )

                        assert not np.isnan(loss), "loss is nan"

                epoch_seconds = time.time() - epoch_start
                logger.info("avg. loss %.3f", loss_meter.avg)
                save_checkpoint(epoch + 1)

                if wandb_enabled:
                    train_results = {
                        "loss": float(loss_meter.avg),
                        "loss_min": float(loss_meter.min),
                        "loss_max": float(loss_meter.max),
                        "mask_a": float(maskA_meter.avg),
                        "mask_b": float(maskB_meter.avg),
                        "iter_time_ms": float(time_meter.avg),
                        "epoch_time_s": float(epoch_seconds),
                        "lr": float(epoch_last_lr),
                        "wd": float(epoch_last_wd),
                        "global_batch_size": int(global_batch_size),
                    }
                    update_log_dict("train", train_results, epoch_log, step="epoch")
                    log_epoch_dict(epoch_log, epoch=epoch + 1)
    finally:
        if wandb_enabled:
            finish_wandb()


if __name__ == "__main__":
    raise SystemExit("Use ijepath.train.main entrypoint")
