import logging
import math
import os
import time
from pathlib import Path
from contextlib import nullcontext
import yaml

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
    log_dict,
    log_images_seen_dict,
    save_run_config_to_wandb,
    update_log_dict,
)
from ijepath.utils.distributed import AllReduce, init_distributed
from ijepath.utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger
from ijepath.utils.log_utils import setup_logging
from ijepath.utils.tensors import repeat_interleave_batch

default_training_save_every = 1_000_000
default_step_log_every_images = 0
default_training_log_every = 100_000
default_tune_every_images = 1_000_000
default_perf_debug_log_every_images = 2_048
default_perf_debug_slow_step_ms = 250.0
default_perf_debug_slow_data_wait_ms = 50.0

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


def resolve_step_log_every_images(logging_cfg: dict, total_images_budget: int) -> int:
    raw_value = logging_cfg.get("step_log_every_images", default_step_log_every_images)
    if isinstance(raw_value, bool):
        raise ValueError("logging.step_log_every_images must be int>=0 or float in [0, 1]")
    if isinstance(raw_value, int):
        if int(raw_value) < 0:
            raise ValueError("logging.step_log_every_images must be int>=0 or float in [0, 1]")
        return int(raw_value)
    if isinstance(raw_value, float):
        value = float(raw_value)
        if value < 0.0 or value > 1.0:
            raise ValueError("logging.step_log_every_images must be int>=0 or float in [0, 1]")
        if value == 0.0:
            return 0
        return max(1, int(round(float(total_images_budget) * value)))
    raise ValueError("logging.step_log_every_images must be int>=0 or float in [0, 1]")


def resolve_use_bfloat16(requested_use_bfloat16: bool, cuda_available: bool) -> bool:
    return bool(requested_use_bfloat16 and cuda_available)


def resolve_training_save_every(training_cfg: dict) -> int:
    interval = int(training_cfg.get("save_every", default_training_save_every))
    if interval <= 0:
        raise ValueError("training.save_every must be > 0")
    return interval


def resolve_training_log_every(training_cfg: dict) -> int:
    interval = int(training_cfg.get("log_every", default_training_log_every))
    if interval <= 0:
        raise ValueError("training.log_every must be > 0")
    return interval


def resolve_performance_debug_config(logging_cfg: dict) -> dict[str, float | int | bool]:
    perf_cfg = dict(logging_cfg.get("performance_debug", {}) or {})
    enabled = bool(perf_cfg.get("enable", False))
    log_every_images = int(perf_cfg.get("log_every_images", default_perf_debug_log_every_images))
    slow_step_ms = float(perf_cfg.get("slow_step_ms", default_perf_debug_slow_step_ms))
    slow_data_wait_ms = float(perf_cfg.get("slow_data_wait_ms", default_perf_debug_slow_data_wait_ms))
    if enabled:
        if log_every_images <= 0:
            raise ValueError("logging.performance_debug.log_every_images must be > 0")
        if slow_step_ms <= 0:
            raise ValueError("logging.performance_debug.slow_step_ms must be > 0")
        if slow_data_wait_ms <= 0:
            raise ValueError("logging.performance_debug.slow_data_wait_ms must be > 0")
    return {
        "enable": enabled,
        "log_every_images": log_every_images,
        "slow_step_ms": slow_step_ms,
        "slow_data_wait_ms": slow_data_wait_ms,
    }


def resolve_tune_poll_every_steps(
    *,
    tuning_execution_cfg: dict,
    tune_every_images: int,
    global_batch_size: int,
) -> int:
    raw_value = tuning_execution_cfg.get("poll_every_steps", "auto")
    if raw_value is None:
        raw_value = "auto"

    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in {"", "auto"}:
            if int(tune_every_images) <= 0:
                raise ValueError("tune_every_images must be > 0")
            if int(global_batch_size) <= 0:
                raise ValueError("global_batch_size must be > 0")
            derived = int(round(float(tune_every_images) / float(int(global_batch_size) * 20)))
            return max(1, int(derived))
        try:
            poll_every_steps = int(token)
        except ValueError as exc:
            raise ValueError("tuning.execution.poll_every_steps must be 'auto' or > 0 integer") from exc
        if poll_every_steps <= 0:
            raise ValueError("tuning.execution.poll_every_steps must be 'auto' or > 0 integer")
        return int(poll_every_steps)

    poll_every_steps = int(raw_value)
    if poll_every_steps <= 0:
        raise ValueError("tuning.execution.poll_every_steps must be 'auto' or > 0 integer")
    return int(poll_every_steps)


def resolve_effective_save_every(
    *,
    training_save_every: int,
    tuning_enabled: bool,
    tune_every_images: int,
) -> int:
    if int(training_save_every) <= 0:
        raise ValueError("training_save_every must be > 0")
    if bool(tuning_enabled):
        if int(tune_every_images) <= 0:
            raise ValueError("tune_every_images must be > 0 when tuning is enabled")
        return int(tune_every_images)
    return int(training_save_every)


def should_warn_training_save_every_override(
    *,
    training_save_every_explicit: bool,
    training_save_every: int,
    tuning_enabled: bool,
    tune_every_images: int,
) -> bool:
    return bool(
        training_save_every_explicit
        and tuning_enabled
        and int(training_save_every) != int(tune_every_images)
    )


def resolve_wandb_exit_code_from_outcome(
    *,
    training_completed_successfully: bool,
    fatal_training_exception: BaseException | None,
) -> int:
    if bool(training_completed_successfully):
        return 0
    if isinstance(fatal_training_exception, SystemExit):
        code = fatal_training_exception.code
        if isinstance(code, int) and code != 0:
            return int(code)
    return 1


def resolve_total_images_budget(optimization_cfg: dict) -> int:
    total_images_budget = optimization_cfg.get("total_images_budget", None)
    if total_images_budget is None:
        raise ValueError("Missing required config value: optimization.total_images_budget")
    if int(total_images_budget) <= 0:
        raise ValueError("optimization.total_images_budget must be > 0")
    return int(total_images_budget)


def compute_total_steps(total_images_budget: int, global_batch_size: int) -> int:
    if int(total_images_budget) <= 0:
        raise ValueError("total_images_budget must be > 0")
    if int(global_batch_size) <= 0:
        raise ValueError("global_batch_size must be > 0")
    return int(math.ceil(float(total_images_budget) / float(global_batch_size)))


def compute_schedule_total_steps(total_steps: int, ipe_scale: float) -> int:
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be > 0")
    if float(ipe_scale) <= 0:
        raise ValueError("ipe_scale must be > 0")
    scaled = int(float(ipe_scale) * float(total_steps))
    return max(int(total_steps), int(scaled))


def compute_total_passes(total_steps: int, steps_per_pass: int) -> int:
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be > 0")
    if int(steps_per_pass) <= 0:
        raise ValueError("steps_per_pass must be > 0")
    return int(math.ceil(float(total_steps) / float(steps_per_pass)))


def build_pass_progress_desc(pass_index: int, total_passes: int) -> str:
    if int(pass_index) <= 0:
        raise ValueError("pass_index must be > 0")
    if int(total_passes) <= 0:
        raise ValueError("total_passes must be > 0")
    return f"Pass [{int(pass_index)}/{int(total_passes)}]"


def _config_file_has_key(path: str | None, section: str, key: str) -> bool:
    if not path:
        return False
    config_path = Path(path)
    if not config_path.exists():
        return False
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    section_payload = payload.get(section, {})
    return isinstance(section_payload, dict) and key in section_payload


def is_training_save_every_explicitly_set(cfg: dict) -> bool:
    config_sources = dict(cfg.get("_config_sources", {}) or {})
    opts = [str(opt) for opt in (config_sources.get("opts") or [])]
    if any(opt.startswith("training.save_every=") for opt in opts):
        return True
    mode = str(config_sources.get("mode", "")).strip().lower()
    if mode == "single":
        return _config_file_has_key(config_sources.get("config_file"), "training", "save_every")
    if mode == "layered":
        return any(
            _config_file_has_key(config_sources.get(path_key), "training", "save_every")
            for path_key in ("run_config", "profile_config")
        )
    return False


def crossed_image_thresholds(
    *,
    prev_images_seen: int,
    new_images_seen: int,
    next_threshold: int,
    interval: int,
) -> tuple[list[int], int]:
    crossed: list[int] = []
    threshold = int(next_threshold)
    while threshold <= int(new_images_seen):
        if threshold > int(prev_images_seen):
            crossed.append(int(threshold))
        threshold += int(interval)
    return crossed, int(threshold)


def checkpoint_name_for_images(tag: str, images_seen: int) -> str:
    return f"{str(tag)}-img{int(images_seen)}.pth.tar"


def compute_anchor_pass_budget(
    *,
    anchor_count: int,
    total_images_budget: int,
    interval_images: int,
    run_baseline_at_zero: bool,
) -> dict[str, float | int]:
    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be > 0")
    if int(total_images_budget) <= 0:
        raise ValueError("total_images_budget must be > 0")
    if int(interval_images) <= 0:
        raise ValueError("interval_images must be > 0")

    anchor_passes_total = float(total_images_budget) / float(anchor_count)
    coverage_first_pass = min(1.0, anchor_passes_total)
    mean_anchor_reuse = max(0.0, anchor_passes_total - 1.0)
    expected_tune_events = int(total_images_budget // interval_images)
    if run_baseline_at_zero:
        expected_tune_events += 1

    return {
        "anchor_count": int(anchor_count),
        "anchor_passes_total": float(anchor_passes_total),
        "coverage_first_pass": float(coverage_first_pass),
        "mean_anchor_reuse": float(mean_anchor_reuse),
        "expected_tune_events": int(expected_tune_events),
    }


def build_pass_train_results(
    *,
    loss_avg: float,
    loss_min: float,
    loss_max: float,
    context_keep_tokens_avg: float,
    target_predict_tokens_avg: float,
    iter_time_ms: float,
    pass_time_s: float,
    images_seen: int,
    anchor_passes_seen: float,
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
        "context_keep_tokens_avg": float(context_keep_tokens_avg),
        "target_predict_tokens_avg": float(target_predict_tokens_avg),
        "iter_time_ms": float(iter_time_ms),
        "pass_time_s": float(pass_time_s),
        "images_seen": int(images_seen),
        "anchor_passes_seen": float(anchor_passes_seen),
        "images_per_sec": float(images_per_sec),
        "iterations_per_sec": float(iterations_per_sec),
        "lr": float(lr),
        "wd": float(wd),
        "global_batch_size": int(global_batch_size),
    }


def get_train_step_csv_columns() -> tuple[tuple[str, str], ...]:
    return (
        ("%d", "images_seen"),
        ("%d", "pass_index"),
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
    images_seen: int,
    pass_index: int,
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
        f"images_seen={int(images_seen)} pass_index={int(pass_index)} iteration={int(iteration)} "
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
    images_seen: int,
    pass_index: int,
    iteration: int,
    first_layer_grad_norm: float,
    last_layer_grad_norm: float,
    grad_min: float,
    grad_max: float,
) -> str:
    return (
        f"images_seen={int(images_seen)} pass_index={int(pass_index)} iteration={int(iteration)} "
        f"first_layer_grad_norm={float(first_layer_grad_norm):.2e} "
        f"last_layer_grad_norm={float(last_layer_grad_norm):.2e} "
        f"grad_norm_min={float(grad_min):.2e} "
        f"grad_norm_max={float(grad_max):.2e}"
    )


def validate_encoder_input_sizes(
    *,
    context_images: torch.Tensor,
    target_images: torch.Tensor,
    expected_context_input_size_px: int,
    expected_target_input_size_px: int,
    patch_size: int,
) -> None:
    if context_images.ndim != 4:
        raise ValueError(
            "context_images must be rank-4 [B, C, H, W], "
            f"got shape={tuple(context_images.shape)}"
        )
    if target_images.ndim != 5:
        raise ValueError(
            "target_images must be rank-5 [B, K, C, H, W], "
            f"got shape={tuple(target_images.shape)}"
        )

    context_h = int(context_images.shape[-2])
    context_w = int(context_images.shape[-1])
    target_h = int(target_images.shape[-2])
    target_w = int(target_images.shape[-1])
    expected_context = int(expected_context_input_size_px)
    expected_target = int(expected_target_input_size_px)
    patch = int(patch_size)

    if context_h != expected_context or context_w != expected_context:
        raise ValueError(
            "Context image size mismatch: "
            f"expected={expected_context}x{expected_context} "
            f"got={context_h}x{context_w}"
        )
    if target_h != expected_target or target_w != expected_target:
        raise ValueError(
            "Target image size mismatch: "
            f"expected={expected_target}x{expected_target} "
            f"got={target_h}x{target_w}"
        )
    if expected_context % patch != 0:
        raise ValueError(
            "Expected context input size must be divisible by patch size: "
            f"context={expected_context} patch={patch}"
        )
    if expected_target % patch != 0:
        raise ValueError(
            "Expected target input size must be divisible by patch size: "
            f"target={expected_target} patch={patch}"
        )
    if context_h % patch != 0 or context_w % patch != 0:
        raise ValueError(
            "Context batch tensor shape is not divisible by patch size: "
            f"context={context_h}x{context_w} patch={patch}"
        )
    if target_h % patch != 0 or target_w % patch != 0:
        raise ValueError(
            "Target batch tensor shape is not divisible by patch size: "
            f"target={target_h}x{target_w} patch={patch}"
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
        logger.warning("WARNING! Disabled bfloat16 AMP because CUDA is unavailable in this runtime.")

    # -- DATA
    batch_size_per_gpu = int(args["data"].get("batch_size_per_gpu", args["data"].get("batch_size")))
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]

    anchor_catalog_manifest = args["data"]["anchor_catalog_manifest"]
    context_mpp = float(args["data"]["context_mpp"])
    target_mpp = float(args["data"]["target_mpp"])
    context_fov_um = float(args["data"]["context_fov_um"])
    target_fov_um = float(args["data"]["target_fov_um"])
    targets_per_context = int(args["data"]["targets_per_context"])
    seed = int(args["data"]["seed"])
    spacing_tolerance = float(args["data"].get("spacing_tolerance", 0.05))
    min_target_tissue_fraction = float(
        args["data"].get("min_target_tissue_fraction", args["data"].get("min_tissue_fraction", 0.25))
    )
    insufficient_target_policy = str(args["data"].get("insufficient_target_policy", "skip_anchor"))
    min_target_tissue_fraction_floor = args["data"].get("min_target_tissue_fraction_floor", None)
    if min_target_tissue_fraction_floor is not None:
        min_target_tissue_fraction_floor = float(min_target_tissue_fraction_floor)
    min_target_tissue_fraction_step = float(args["data"].get("min_target_tissue_fraction_step", 0.05))
    align_targets_to_patch_grid = args["data"].get("align_targets_to_patch_grid", False)
    sampling_strategy = str(args["data"].get("sampling_strategy", "stratified_weighted"))
    sampling_stratum_key = str(args["data"].get("sampling_stratum_key", "organ"))
    sampling_stratum_weights = args["data"].get("sampling_stratum_weights", "inverse_frequency")
    persistent_workers = bool(args["data"].get("persistent_workers", True))
    prefetch_factor = int(args["data"].get("prefetch_factor", 4))
    max_open_slides_per_worker = int(args["data"].get("max_open_slides_per_worker", 16))
    anchor_stream_batch_size = int(args["data"].get("anchor_stream_batch_size", 2048))
    wsi_backend = str(args["data"].get("wsi_backend", "asap"))
    low_anchor_pass_warning_threshold = float(args["data"].get("low_anchor_pass_warning_threshold", 1.0))
    high_anchor_pass_warning_threshold = float(args["data"].get("high_anchor_pass_warning_threshold", 5.0))
    if low_anchor_pass_warning_threshold <= 0:
        raise ValueError("data.low_anchor_pass_warning_threshold must be > 0")
    if high_anchor_pass_warning_threshold <= 0:
        raise ValueError("data.high_anchor_pass_warning_threshold must be > 0")
    if high_anchor_pass_warning_threshold <= low_anchor_pass_warning_threshold:
        raise ValueError(
            "data.high_anchor_pass_warning_threshold must be > "
            "data.low_anchor_pass_warning_threshold"
        )

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
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]
    total_images_budget = resolve_total_images_budget(args["optimization"])

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    step_log_every_images_raw = args["logging"].get(
        "step_log_every_images",
        default_step_log_every_images,
    )
    step_log_every_images = resolve_step_log_every_images(
        args["logging"],
        total_images_budget=int(total_images_budget),
    )
    expected_step_log_events = (
        int(math.ceil(float(total_images_budget) / float(step_log_every_images)))
        if step_log_every_images > 0
        else 0
    )
    training_cfg = dict(args.get("training", {}) or {})
    training_save_every = resolve_training_save_every(training_cfg)
    training_log_every = resolve_training_log_every(training_cfg)
    perf_debug_cfg = resolve_performance_debug_config(args.get("logging", {}))
    perf_debug_enabled = bool(perf_debug_cfg["enable"])
    perf_debug_log_every_images = int(perf_debug_cfg["log_every_images"])
    perf_debug_slow_step_ms = float(perf_debug_cfg["slow_step_ms"])
    perf_debug_slow_data_wait_ms = float(perf_debug_cfg["slow_data_wait_ms"])

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
    total_steps = compute_total_steps(
        total_images_budget=total_images_budget,
        global_batch_size=global_batch_size,
    )

    args.setdefault("data", {})
    args["data"]["batch_size_per_gpu"] = batch_size_per_gpu
    args["data"]["global_batch_size"] = global_batch_size
    args["data"]["world_size"] = world_size
    args["data"]["context_input_size_px"] = context_input_size_px
    args["data"]["target_input_size_px"] = target_input_size_px
    args["data"]["anchor_stream_batch_size"] = anchor_stream_batch_size
    args.setdefault("logging", {})
    args["logging"]["performance_debug"] = {
        "enable": perf_debug_enabled,
        "log_every_images": perf_debug_log_every_images,
        "slow_step_ms": perf_debug_slow_step_ms,
        "slow_data_wait_ms": perf_debug_slow_data_wait_ms,
    }
    args.setdefault("optimization", {})
    args["optimization"]["total_steps"] = total_steps

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
        f"Batch sizing:\n"
        f" - batch_size_per_gpu={batch_size_per_gpu}\n"
        f" - global_batch_size={global_batch_size}"
    )
    logger.info(
        "Step logging cadence:\n"
        f" - logging.step_log_every_images(raw)={step_log_every_images_raw!r}\n"
        f" - logging.step_log_every_images(resolved)={step_log_every_images:,}\n"
        f" - expected_step_log_events={expected_step_log_events:,}"
    )
    logger.info(
        "Image-budget control:\n"
        f" - total_images_budget={total_images_budget:,}\n"
        f" - total_steps={total_steps:,}\n"
        f" - training.save_every={training_save_every:,}"
    )
    logger.info(
        "Train logging cadence:\n"
        f" - training.log_every={training_log_every:,}"
    )
    if perf_debug_enabled:
        logger.info(
            "Performance debug logging enabled:\n"
            f" - log_every_images={perf_debug_log_every_images:,}\n"
            f" - slow_step_ms={perf_debug_slow_step_ms:.1f}\n"
            f" - slow_data_wait_ms={perf_debug_slow_data_wait_ms:.1f}"
        )

    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
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
    target_encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=target_input_size_px,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        architecture=architecture,
        init_predictor=False,
    )
    matched_params, skipped_params = copy_matching_state_dict_params(
        source=encoder,
        target=target_encoder,
    )
    logger.info(
        "Initialized target encoder from context encoder:\n"
        f" - matched_state_entries={matched_params}\n"
        f" - skipped_state_entries={skipped_params}"
    )

    unsupervised_dataset, unsupervised_loader, unsupervised_sampler = make_cross_resolution_loader(
        batch_size=batch_size_per_gpu,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        anchor_catalog_manifest=anchor_catalog_manifest,
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
        align_targets_to_patch_grid=align_targets_to_patch_grid,
        sampling_strategy=sampling_strategy,
        sampling_stratum_key=sampling_stratum_key,
        sampling_stratum_weights=sampling_stratum_weights,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        max_open_slides_per_worker=max_open_slides_per_worker,
        anchor_stream_batch_size=anchor_stream_batch_size,
    )
    ipe = len(unsupervised_loader)
    if ipe <= 0:
        raise ValueError("Unsupervised loader produced zero iterations per anchor pass")

    schedule_total_steps = compute_schedule_total_steps(
        total_steps=total_steps,
        ipe_scale=ipe_scale,
    )
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        total_steps=total_steps,
        warmup=warmup,
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
        ema[0] + i * (ema[1] - ema[0]) / schedule_total_steps
        for i in range(int(schedule_total_steps) + 1)
    )

    start_pass_index = 0
    resumed_steps_done = 0
    resumed_images_seen: int | None = None
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_pass_index = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        resumed_steps_done = int(start_pass_index) * int(ipe)
        try:
            if os.path.exists(load_path):
                ckpt_meta = torch.load(load_path, map_location=torch.device("cpu"))
                resumed_steps_done = int(ckpt_meta.get("steps_done", resumed_steps_done))
                if ckpt_meta.get("images_seen") is not None:
                    resumed_images_seen = int(ckpt_meta.get("images_seen"))
        except Exception as exc:
            logger.warning("WARNING! Failed to read checkpoint metadata for step resume info: %s", exc)

    resumed_steps_done = min(int(resumed_steps_done), int(total_steps))
    for _ in range(resumed_steps_done):
        scheduler.step()
        wd_scheduler.step()
        next(momentum_scheduler)

    steps_done = int(resumed_steps_done)
    images_seen = int(
        min(
            int(total_images_budget),
            int(resumed_images_seen if resumed_images_seen is not None else steps_done * global_batch_size),
        )
    )
    pass_index = int(start_pass_index)

    def build_snapshot(*, pass_id: int, loss_avg: float) -> dict:
        return {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "pass_index": int(pass_id),
            "steps_done": int(steps_done),
            "images_seen": int(images_seen),
            "loss": float(loss_avg),
            "batch_size_per_gpu": batch_size_per_gpu,
            "global_batch_size": global_batch_size,
            "world_size": world_size,
            "lr": lr,
        }

    def save_snapshot(snapshot: dict, *, images_tag: int | None = None, write_latest: bool = True) -> None:
        if rank != 0:
            return
        if write_latest:
            torch.save(snapshot, latest_path)
        if images_tag is not None:
            image_ckpt_path = os.path.join(folder, checkpoint_name_for_images(tag=tag, images_seen=images_tag))
            torch.save(snapshot, image_ckpt_path)

    tuning_cfg = dict(args.get("tuning", {}) or {})
    tuning_enabled_cfg = bool(tuning_cfg.get("enable", False))
    tune_every_images = int(tuning_cfg.get("tune_every", default_tune_every_images))
    if tuning_enabled_cfg and tune_every_images <= 0:
        raise ValueError("tuning.tune_every must be > 0")
    run_baseline_at_zero = bool(tuning_cfg.get("run_baseline_at_zero", True))
    training_save_every_explicit = is_training_save_every_explicitly_set(args)
    effective_save_every = resolve_effective_save_every(
        training_save_every=int(training_save_every),
        tuning_enabled=bool(tuning_enabled_cfg),
        tune_every_images=int(tune_every_images),
    )
    if should_warn_training_save_every_override(
        training_save_every_explicit=training_save_every_explicit,
        training_save_every=int(training_save_every),
        tuning_enabled=bool(tuning_enabled_cfg),
        tune_every_images=int(tune_every_images),
    ):
        logger.warning(
            "WARNING! tuning.enable=true overrides training.save_every=%d with tuning.tune_every=%d for checkpoint cadence.",
            int(training_save_every),
            int(tune_every_images),
        )
    anchor_count = int(
        getattr(unsupervised_dataset, "total_anchors", 0)
        or len(getattr(unsupervised_dataset, "anchors", []))
        or len(unsupervised_dataset)
    )
    anchor_budget = compute_anchor_pass_budget(
        anchor_count=anchor_count,
        total_images_budget=total_images_budget,
        interval_images=max(1, tune_every_images),
        run_baseline_at_zero=run_baseline_at_zero,
    )
    if not tuning_enabled_cfg:
        anchor_budget["expected_tune_events"] = 0
    logger.info(
        "Anchor diversity budget:\n"
        f" - anchor_count={int(anchor_budget['anchor_count']):,}\n"
        f" - total_images_budget={int(total_images_budget):,}\n"
        f" - coverage_first_pass={float(anchor_budget['coverage_first_pass']):.2f}\n"
        f" - mean_anchor_reuse={float(anchor_budget['mean_anchor_reuse']):.2f}\n"
        f" - expected_tune_events={int(anchor_budget['expected_tune_events']):,}"
    )
    if float(anchor_budget["anchor_passes_total"]) < float(low_anchor_pass_warning_threshold):
        logger.warning(
            f"WARNING! Anchor pass budget is low (anchor_passes_total={float(anchor_budget['anchor_passes_total']):.2f} < {low_anchor_pass_warning_threshold}). "
            "Increase total_images_budget or reduce anchor catalog size to improve first-pass coverage. "
            "For extra target-placement diversity per anchor, increase data.targets_per_context."
        )
    if float(anchor_budget["anchor_passes_total"]) > float(high_anchor_pass_warning_threshold):
        logger.warning(
            f"WARNING! Anchor pass budget is high (anchor_passes_total={float(anchor_budget['anchor_passes_total']):.2f} > {high_anchor_pass_warning_threshold}). "
            "Anchor reuse may dominate context diversity. Increase anchor catalog diversity (add slides) or reduce total_images_budget. "
            "For extra target-placement diversity per anchor, increase data.targets_per_context."
        )

    total_passes = compute_total_passes(total_steps=total_steps, steps_per_pass=ipe)
    tuning_execution_cfg = dict(tuning_cfg.get("execution", {}) or {})
    tune_poll_every_steps = resolve_tune_poll_every_steps(
        tuning_execution_cfg=tuning_execution_cfg,
        tune_every_images=int(tune_every_images),
        global_batch_size=int(global_batch_size),
    )
    expected_train_log_events = int(math.ceil(float(total_images_budget) / float(training_log_every)))
    logger.info(
        "Cadence vs image budget:\n"
        f" - training.log_every={training_log_every:,} (~{expected_train_log_events:,} train logs over budget)\n"
        f" - tuning.tune_every={tune_every_images:,} "
        f"(enabled={tuning_enabled_cfg}, expected_tune_events={int(anchor_budget['expected_tune_events']):,})\n"
        f" - tuning.execution.poll_every_steps={int(tune_poll_every_steps):,}\n"
        f" - effective_checkpoint_save_every={effective_save_every:,}"
    )
    logger.info(
        "Training pass plan:\n"
        f" - total_passes={total_passes:,} (progress bar shows Pass [x/{total_passes}])\n"
        f" - anchor_passes_total={float(anchor_budget['anchor_passes_total']):.4f}"
    )

    tuner = None
    tuning_runtime = None
    early_stopper = None
    early_stop_cfg = dict(tuning_cfg.get("early_stopping", {}) or {})
    pending_tune_snapshots: dict[int, dict] = {}
    tuning_output_dir = Path(folder) / "tuning"
    tuning_snapshots_dir = tuning_output_dir / "snapshots"
    tuning_snapshots_dir.mkdir(parents=True, exist_ok=True)

    if tuning_enabled_cfg and rank == 0:
        from ijepath.eval.async_runtime import AsyncTuningRuntime
        from ijepath.eval.tuner import Tuner

        teacher_backbone = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
        tuner = Tuner(
            cfg=tuning_cfg,
            device=device,
            output_dir=tuning_output_dir,
        )
        tuning_runtime = AsyncTuningRuntime(
            tuning_cfg=tuning_cfg,
            teacher_template=teacher_backbone,
            output_dir=tuning_output_dir,
        )

        if bool(early_stop_cfg.get("enable", False)):
            from ijepath.eval.early_stopping import RobustnessEarlyStopper

            selected_plugin_mode = str(tuner.get_selection_mode() or "max")

            early_stopper = RobustnessEarlyStopper(
                mode=selected_plugin_mode,
                patience_evals=int(early_stop_cfg.get("patience_evals", 5)),
                min_evals=int(early_stop_cfg.get("min_evals", 3)),
                checkpoint_path=Path(folder) / str(
                    early_stop_cfg.get("best_checkpoint_name", "best-robustness.pth.tar")
                ),
                save_best_checkpoint=bool(early_stop_cfg.get("save_best_checkpoint", True)),
            )

    next_checkpoint_images = int((images_seen // effective_save_every) + 1) * int(effective_save_every)
    next_wandb_log_images = int((images_seen // training_log_every) + 1) * int(training_log_every)
    if step_log_every_images > 0:
        next_step_log_images = int((images_seen // step_log_every_images) + 1) * int(step_log_every_images)
    else:
        next_step_log_images = 0
    if tuning_enabled_cfg:
        next_tune_images = int((images_seen // tune_every_images) + 1) * int(tune_every_images)
    else:
        next_tune_images = 0
    tune_index = int(images_seen // tune_every_images) if tuning_enabled_cfg else 0
    if tuning_enabled_cfg and run_baseline_at_zero and images_seen > 0:
        tune_index += 1

    wandb_loss_meter = AverageMeter()
    wandb_maskA_meter = AverageMeter()
    wandb_maskB_meter = AverageMeter()
    wandb_time_meter = AverageMeter()
    wandb_window_images = 0
    wandb_window_iters = 0
    wandb_window_seconds = 0.0
    wandb_window_last_lr = float("nan")
    wandb_window_last_wd = float("nan")
    last_wandb_train_log_images = int(images_seen)

    def enqueue_tuning_job(
        *,
        tune_index_value: int,
        tune_images_seen: int,
        early_stop_snapshot: dict,
        images_seen_at_log: int,
    ) -> None:
        if rank != 0 or tuning_runtime is None:
            return

        teacher_backbone = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
        snapshot_path = tuning_snapshots_dir / f"teacher_tune_{int(tune_index_value):06d}.pth"
        torch.save(teacher_backbone.state_dict(), snapshot_path)
        queue_stats = tuning_runtime.submit(
            tune_index=int(tune_index_value),
            images_seen=int(tune_images_seen),
            snapshot_path=str(snapshot_path),
        )
        pending_tune_snapshots[int(tune_index_value)] = dict(early_stop_snapshot)

        if wandb_enabled:
            queue_log: dict[str, float | int] = {
                "images_seen": int(tune_images_seen),
                "train/images_seen_at_log": int(images_seen_at_log),
            }
            queue_metrics = {
                "tune_index": int(tune_index_value),
                "tune_images_seen": int(tune_images_seen),
                "queue_depth": int(queue_stats.get("queue_depth", 0)),
                "dropped_tunes": int(queue_stats.get("dropped_tunes", 0)),
            }
            update_log_dict("tune", queue_metrics, queue_log, step="images_seen")
            log_dict(queue_log)

    def process_tuning_completions(*, images_seen_at_log: int) -> None:
        nonlocal should_stop_training
        if rank != 0 or tuning_runtime is None:
            return

        completed = tuning_runtime.poll_completed()
        for event in completed:
            if event.get("error") is not None:
                raise RuntimeError(
                    "Async tuning run failed "
                    f"(tune_index={event.get('tune_index')}): {event.get('error')}\n"
                    f"{event.get('traceback') or ''}"
                )

            tune_results = dict(event.get("result") or {})
            tune_metrics = dict(tune_results.get("log_metrics", {}) or {})
            event_tune_index = int(event.get("tune_index", -1))
            event_images_seen = int(event.get("images_seen", 0))
            event_latency = float(event.get("latency_seconds", 0.0))
            event_worker_seconds = float(event.get("worker_seconds", 0.0))

            if tune_metrics:
                logger.info(
                    "Async tuning results at images_seen=%d (tune_index=%d): %s",
                    event_images_seen,
                    event_tune_index,
                    ", ".join(f"{k}={v:.5f}" for k, v in sorted(tune_metrics.items())),
                )

            if wandb_enabled:
                tune_log: dict[str, float | int] = {
                    "images_seen": int(event_images_seen),
                    "train/images_seen_at_log": int(images_seen_at_log),
                }
                tune_infra_metrics = {
                    "tune_index": int(event_tune_index),
                    "tune_images_seen": int(event_images_seen),
                    "queue_depth": int(tuning_runtime.queue_depth()),
                    "dropped_tunes": int(tuning_runtime.dropped_tunes()),
                    "result_lag_images": int(max(0, int(images_seen_at_log) - int(event_images_seen))),
                    "result_lag_seconds": float(event_latency),
                    "worker_seconds": float(event_worker_seconds),
                    "tuning_blocked_ms": 0.0,
                }
                update_log_dict("tune", tune_infra_metrics, tune_log, step="images_seen")
                if tune_metrics:
                    update_log_dict("tune", tune_metrics, tune_log, step="images_seen")
                log_dict(tune_log)

            snapshot = pending_tune_snapshots.pop(event_tune_index, None)
            if early_stopper is not None:
                selection = tune_results.get("selection")
                if selection is not None and selection.get("metric_value") is not None:
                    if snapshot is None:
                        snapshot = build_snapshot(
                            pass_id=max(0, pass_index),
                            loss_avg=float("nan"),
                        )
                    early_stopper.on_tune(
                        metric_value=float(selection["metric_value"]),
                        snapshot=snapshot,
                    )
                    if bool(early_stop_cfg.get("stop_training", False)) and early_stopper.should_stop:
                        should_stop_training = True

    def sync_stop_flag(local_flag: bool) -> bool:
        if not use_ddp:
            return bool(local_flag)
        flag_tensor = torch.tensor([1 if local_flag else 0], device=device, dtype=torch.int32)
        dist.broadcast(flag_tensor, src=0)
        return bool(int(flag_tensor.item()))

    should_stop_training = False
    if perf_debug_enabled:
        perf_data_wait_meter = AverageMeter()
        perf_h2d_meter = AverageMeter()
        perf_compute_meter = AverageMeter()
        perf_tune_poll_meter = AverageMeter()
        perf_overhead_meter = AverageMeter()
        perf_loop_meter = AverageMeter()
        perf_cache_open_meter = AverageMeter()
        perf_cache_hit_count = 0
        perf_cache_miss_count = 0
        perf_cache_eviction_count = 0
        perf_cache_sample_count = 0
        perf_next_log_images = int(perf_debug_log_every_images)
    training_completed_successfully = False
    fatal_training_exception: BaseException | None = None
    try:
        if tuning_enabled_cfg and run_baseline_at_zero and images_seen == 0:
            if rank == 0 and tuning_runtime is not None:
                enqueue_tuning_job(
                    tune_index_value=0,
                    tune_images_seen=0,
                    early_stop_snapshot=build_snapshot(pass_id=0, loss_avg=float("nan")),
                    images_seen_at_log=0,
                )
            tune_index = max(tune_index, 1)

        while steps_done < total_steps and not should_stop_training:
            pass_start = time.time()
            current_pass_index = int(pass_index + 1)
            if hasattr(unsupervised_dataset, "set_pass_index"):
                unsupervised_dataset.set_pass_index(pass_index)
            if unsupervised_sampler is not None and hasattr(unsupervised_sampler, "set_epoch"):
                unsupervised_sampler.set_epoch(pass_index)

            loss_meter = AverageMeter()
            maskA_meter = AverageMeter()
            maskB_meter = AverageMeter()
            time_meter = AverageMeter()
            pass_last_lr = float("nan")
            pass_last_wd = float("nan")
            pass_images_start = int(images_seen)
            pass_iters_done = 0

            remaining_steps = int(total_steps - steps_done)
            pass_target_images = int(min(ipe, remaining_steps) * global_batch_size)

            with tqdm.tqdm(
                total=pass_target_images,
                desc=build_pass_progress_desc(
                    pass_index=current_pass_index,
                    total_passes=total_passes,
                ),
                unit=" img",
                unit_scale=True,
                ncols=100,
                leave=False,
                disable=rank != 0,
            ) as iter_bar:
                loader_iter = iter(unsupervised_loader)
                for itr in range(ipe):
                    if steps_done >= total_steps or should_stop_training:
                        break
                    iter_wall_start = time.perf_counter()
                    data_wait_start = time.perf_counter()
                    try:
                        batch_data, masks_enc, masks_pred = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(unsupervised_loader)
                        batch_data, masks_enc, masks_pred = next(loader_iter)
                    data_wait_ms = float((time.perf_counter() - data_wait_start) * 1_000.0)
                    sample_metadata = batch_data.get("sample_metadata", [])
                    batch_cache_hits = int(
                        sum(int(meta.get("reader_cache_hit", 0)) for meta in sample_metadata)
                    )
                    batch_cache_misses = int(
                        sum(int(meta.get("reader_cache_miss", 0)) for meta in sample_metadata)
                    )
                    batch_cache_evictions = int(
                        sum(int(meta.get("reader_cache_evictions_on_event", 0)) for meta in sample_metadata)
                    )
                    batch_cache_samples = int(len(sample_metadata))
                    batch_cache_miss_ratio = float(
                        float(batch_cache_misses) / float(max(1, batch_cache_samples))
                    )

                    host_to_device_start = time.perf_counter()
                    context_imgs = batch_data["context_images"].to(device, non_blocking=True)
                    target_imgs = batch_data["target_images"].to(device, non_blocking=True)
                    masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
                    masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
                    host_to_device_ms = float((time.perf_counter() - host_to_device_start) * 1_000.0)
                    validate_encoder_input_sizes(
                        context_images=context_imgs,
                        target_images=target_imgs,
                        expected_context_input_size_px=context_input_size_px,
                        expected_target_input_size_px=target_input_size_px,
                        patch_size=patch_size,
                    )

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
                    pass_last_lr = float(_new_lr)
                    pass_last_wd = float(_new_wd)

                    steps_done += 1
                    pass_iters_done += 1
                    prev_images_seen = int(images_seen)
                    images_seen = int(min(total_images_budget, prev_images_seen + global_batch_size))
                    iter_images_processed = int(
                        min(global_batch_size, max(0, total_images_budget - prev_images_seen))
                    )

                    loss_meter.update(loss)
                    time_meter.update(etime)

                    csv_logger.log(
                        images_seen,
                        current_pass_index,
                        itr,
                        loss,
                        maskA_meter.val,
                        maskB_meter.val,
                        etime,
                        pass_last_lr,
                        pass_last_wd,
                    )

                    if rank == 0:
                        iter_bar.update(iter_images_processed)
                        iter_bar.set_postfix(
                            loss=f"{loss_meter.avg:.3f}",
                            lr=f"{pass_last_lr:.2e}",
                            wd=f"{pass_last_wd:.2e}",
                        )

                    if wandb_enabled:
                        wandb_loss_meter.update(loss)
                        wandb_maskA_meter.update(maskA_meter.val)
                        wandb_maskB_meter.update(maskB_meter.val)
                        wandb_time_meter.update(etime)
                        wandb_window_images += int(iter_images_processed)
                        wandb_window_iters += 1
                        wandb_window_last_lr = float(pass_last_lr)
                        wandb_window_last_wd = float(pass_last_wd)
                        wandb_window_seconds += max(0.0, float(time.time() - iter_wall_start))

                    crossed_step_logs = []
                    if step_log_every_images > 0:
                        crossed_step_logs, next_step_log_images = crossed_image_thresholds(
                            prev_images_seen=prev_images_seen,
                            new_images_seen=images_seen,
                            next_threshold=next_step_log_images,
                            interval=step_log_every_images,
                        )
                    if crossed_step_logs or np.isnan(loss) or np.isinf(loss):
                        if torch.cuda.is_available():
                            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0**2
                        else:
                            max_mem_mb = 0.0
                        logger.info(
                            build_step_log_line(
                                images_seen=int(images_seen),
                                pass_index=current_pass_index,
                                iteration=itr,
                                loss_avg=float(loss_meter.avg),
                                context_keep_tokens=float(maskA_meter.avg),
                                target_predict_tokens=float(maskB_meter.avg),
                                weight_decay=float(pass_last_wd),
                                learning_rate=float(pass_last_lr),
                                max_memory_mb=float(max_mem_mb),
                                iteration_time_ms=float(time_meter.avg),
                            )
                        )
                        if grad_stats is not None:
                            logger.info(
                                build_grad_stats_log_line(
                                    images_seen=int(images_seen),
                                    pass_index=current_pass_index,
                                    iteration=itr,
                                    first_layer_grad_norm=float(grad_stats.first_layer),
                                    last_layer_grad_norm=float(grad_stats.last_layer),
                                    grad_min=float(grad_stats.min),
                                    grad_max=float(grad_stats.max),
                                )
                            )

                    assert not np.isnan(loss), "loss is nan"

                    crossed_ckpt, next_checkpoint_images = crossed_image_thresholds(
                        prev_images_seen=prev_images_seen,
                        new_images_seen=images_seen,
                        next_threshold=next_checkpoint_images,
                        interval=effective_save_every,
                    )
                    for crossed in crossed_ckpt:
                        save_snapshot(
                            build_snapshot(pass_id=current_pass_index, loss_avg=float(loss_meter.avg)),
                            images_tag=int(crossed),
                            write_latest=True,
                        )

                    crossed_wandb, next_wandb_log_images = crossed_image_thresholds(
                        prev_images_seen=prev_images_seen,
                        new_images_seen=images_seen,
                        next_threshold=next_wandb_log_images,
                        interval=training_log_every,
                    )
                    if wandb_enabled and crossed_wandb and wandb_loss_meter.count > 0:
                        window_seconds = float(max(0.0, wandb_window_seconds))
                        window_images = int(max(0, wandb_window_images))
                        window_iters = int(max(0, wandb_window_iters))
                        window_images_per_sec = (
                            float(window_images) / window_seconds
                            if window_seconds > 0.0
                            else 0.0
                        )
                        window_iters_per_sec = (
                            float(window_iters) / window_seconds
                            if window_seconds > 0.0
                            else 0.0
                        )
                        for wandb_log_step in crossed_wandb:
                            train_results = build_pass_train_results(
                                loss_avg=float(wandb_loss_meter.avg),
                                loss_min=float(wandb_loss_meter.min),
                                loss_max=float(wandb_loss_meter.max),
                                context_keep_tokens_avg=float(wandb_maskA_meter.avg),
                                target_predict_tokens_avg=float(wandb_maskB_meter.avg),
                                iter_time_ms=float(wandb_time_meter.avg),
                                pass_time_s=float(window_seconds),
                                images_seen=int(wandb_log_step),
                                anchor_passes_seen=float(wandb_log_step) / float(max(1, anchor_count)),
                                images_per_sec=float(window_images_per_sec),
                                iterations_per_sec=float(window_iters_per_sec),
                                lr=float(wandb_window_last_lr),
                                wd=float(wandb_window_last_wd),
                                global_batch_size=int(global_batch_size),
                            )
                            train_log: dict[str, float | int] = {
                                "pass_index": int(current_pass_index),
                                "images_seen": int(wandb_log_step),
                            }
                            update_log_dict("train", train_results, train_log, step="images_seen")
                            log_images_seen_dict(train_log, images_seen=int(wandb_log_step))
                        wandb_loss_meter.reset()
                        wandb_maskA_meter.reset()
                        wandb_maskB_meter.reset()
                        wandb_time_meter.reset()
                        wandb_window_images = 0
                        wandb_window_iters = 0
                        wandb_window_seconds = 0.0
                        last_wandb_train_log_images = int(crossed_wandb[-1])

                    if tuning_enabled_cfg:
                        crossed_tune, next_tune_images = crossed_image_thresholds(
                            prev_images_seen=prev_images_seen,
                            new_images_seen=images_seen,
                            next_threshold=next_tune_images,
                            interval=tune_every_images,
                        )
                    else:
                        crossed_tune = []

                    for crossed in crossed_tune:
                        current_tune_index = int(tune_index)
                        if rank == 0 and tuning_runtime is not None:
                            enqueue_tuning_job(
                                tune_index_value=current_tune_index,
                                tune_images_seen=int(crossed),
                                early_stop_snapshot=build_snapshot(
                                    pass_id=current_pass_index,
                                    loss_avg=float(loss_meter.avg),
                                ),
                                images_seen_at_log=int(images_seen),
                            )
                        tune_index += 1

                    tune_poll_ms = 0.0
                    if tuning_enabled_cfg and rank == 0 and tuning_runtime is not None:
                        if (steps_done % max(1, tune_poll_every_steps)) == 0:
                            tune_poll_start = time.perf_counter()
                            process_tuning_completions(images_seen_at_log=int(images_seen))
                            tune_poll_ms = float((time.perf_counter() - tune_poll_start) * 1_000.0)

                    iter_loop_ms = float((time.perf_counter() - iter_wall_start) * 1_000.0)
                    compute_ms = float(max(0.0, float(etime)))
                    accounted_ms = data_wait_ms + host_to_device_ms + compute_ms + tune_poll_ms
                    overhead_ms = float(max(0.0, iter_loop_ms - accounted_ms))
                    if perf_debug_enabled and rank == 0:
                        perf_data_wait_meter.update(data_wait_ms)
                        perf_h2d_meter.update(host_to_device_ms)
                        perf_compute_meter.update(compute_ms)
                        perf_tune_poll_meter.update(tune_poll_ms)
                        perf_overhead_meter.update(overhead_ms)
                        perf_loop_meter.update(iter_loop_ms)
                        perf_cache_hit_count += int(batch_cache_hits)
                        perf_cache_miss_count += int(batch_cache_misses)
                        perf_cache_eviction_count += int(batch_cache_evictions)
                        perf_cache_sample_count += int(batch_cache_samples)
                        if batch_cache_samples > 0:
                            perf_cache_open_meter.update(
                                float(
                                    sum(float(meta.get("reader_cache_open_slides", 0.0)) for meta in sample_metadata)
                                    / float(batch_cache_samples)
                                )
                            )

                        if (
                            iter_loop_ms >= perf_debug_slow_step_ms
                            or data_wait_ms >= perf_debug_slow_data_wait_ms
                        ):
                            meta0 = sample_metadata[0] if sample_metadata else {}
                            logger.info(
                                "Perf spike at images_seen=%d step=%d: loop_ms=%.1f data_wait_ms=%.1f "
                                "h2d_ms=%.1f compute_ms=%.1f tune_poll_ms=%.1f overhead_ms=%.1f "
                                "cache(hit/miss/evict)=%d/%d/%d miss_ratio=%.2f "
                                "stratum=%s stream_batch=%s row_in_batch=%s",
                                int(images_seen),
                                int(steps_done),
                                float(iter_loop_ms),
                                float(data_wait_ms),
                                float(host_to_device_ms),
                                float(compute_ms),
                                float(tune_poll_ms),
                                float(overhead_ms),
                                int(batch_cache_hits),
                                int(batch_cache_misses),
                                int(batch_cache_evictions),
                                float(batch_cache_miss_ratio),
                                str(meta0.get("stratum_id", "unknown")),
                                str(meta0.get("anchor_stream_batch_id", "n/a")),
                                str(meta0.get("anchor_stream_row_in_batch", "n/a")),
                            )

                        crossed_perf, perf_next_log_images = crossed_image_thresholds(
                            prev_images_seen=prev_images_seen,
                            new_images_seen=images_seen,
                            next_threshold=perf_next_log_images,
                            interval=perf_debug_log_every_images,
                        )
                        if crossed_perf and perf_loop_meter.count > 0:
                            logger.info(
                                "Perf window up to images_seen=%d: loop(avg/max)=%.1f/%.1f ms "
                                "data_wait(avg/max)=%.1f/%.1f ms h2d(avg/max)=%.1f/%.1f ms "
                                "compute(avg/max)=%.1f/%.1f ms tune_poll(avg/max)=%.1f/%.1f ms "
                                "overhead(avg/max)=%.1f/%.1f ms cache_hit_ratio=%.2f "
                                "cache_miss_ratio=%.2f cache_evictions=%d cache_open_slides_avg=%.1f",
                                int(crossed_perf[-1]),
                                float(perf_loop_meter.avg),
                                float(perf_loop_meter.max),
                                float(perf_data_wait_meter.avg),
                                float(perf_data_wait_meter.max),
                                float(perf_h2d_meter.avg),
                                float(perf_h2d_meter.max),
                                float(perf_compute_meter.avg),
                                float(perf_compute_meter.max),
                                float(perf_tune_poll_meter.avg),
                                float(perf_tune_poll_meter.max),
                                float(perf_overhead_meter.avg),
                                float(perf_overhead_meter.max),
                                float(perf_cache_hit_count / max(1, perf_cache_sample_count)),
                                float(perf_cache_miss_count / max(1, perf_cache_sample_count)),
                                int(perf_cache_eviction_count),
                                float(perf_cache_open_meter.avg if perf_cache_open_meter.count > 0 else 0.0),
                            )
                            perf_data_wait_meter.reset()
                            perf_h2d_meter.reset()
                            perf_compute_meter.reset()
                            perf_tune_poll_meter.reset()
                            perf_overhead_meter.reset()
                            perf_loop_meter.reset()
                            perf_cache_open_meter.reset()
                            perf_cache_hit_count = 0
                            perf_cache_miss_count = 0
                            perf_cache_eviction_count = 0
                            perf_cache_sample_count = 0

                    should_stop_training = sync_stop_flag(bool(should_stop_training))

                    if steps_done >= total_steps or should_stop_training:
                        break

            pass_seconds = time.time() - pass_start
            pass_images_processed = int(images_seen - pass_images_start)
            pass_images_per_sec = (
                float(pass_images_processed) / float(pass_seconds)
                if pass_seconds > 0
                else 0.0
            )
            pass_iterations_per_sec = (
                float(pass_iters_done) / float(pass_seconds)
                if pass_seconds > 0
                else 0.0
            )
            anchor_passes_seen = float(images_seen) / float(max(1, anchor_count))

            logger.info(
                f"Pass {current_pass_index} summary: "
                f"loss(avg/min/max)={float(loss_meter.avg):.3f}/{float(loss_meter.min):.3f}/{float(loss_meter.max):.3f} "
                f"lr={float(pass_last_lr):.2e} wd={float(pass_last_wd):.2e} "
                f"images_seen={images_seen}/{total_images_budget} "
                f"anchor_passes_seen={anchor_passes_seen:.4f} "
                f"images_per_sec={pass_images_per_sec:.1f} "
                f"iterations_per_sec={pass_iterations_per_sec:.2f} "
                f"iter_time_ms={float(time_meter.avg):.1f} "
                f"masks(avg)={float(maskA_meter.avg):.1f}/{float(maskB_meter.avg):.1f}"
            )

            save_snapshot(
                build_snapshot(pass_id=current_pass_index, loss_avg=float(loss_meter.avg)),
                images_tag=None,
                write_latest=True,
            )

            pass_index += 1

        if tuning_enabled_cfg and rank == 0 and tuning_runtime is not None:
            process_tuning_completions(images_seen_at_log=int(images_seen))

        if (
            wandb_enabled
            and wandb_loss_meter.count > 0
            and int(images_seen) > int(last_wandb_train_log_images)
        ):
            window_seconds = float(max(0.0, wandb_window_seconds))
            window_images = int(max(0, wandb_window_images))
            window_iters = int(max(0, wandb_window_iters))
            window_images_per_sec = (
                float(window_images) / window_seconds
                if window_seconds > 0.0
                else 0.0
            )
            window_iters_per_sec = (
                float(window_iters) / window_seconds
                if window_seconds > 0.0
                else 0.0
            )
            train_results = build_pass_train_results(
                loss_avg=float(wandb_loss_meter.avg),
                loss_min=float(wandb_loss_meter.min),
                loss_max=float(wandb_loss_meter.max),
                context_keep_tokens_avg=float(wandb_maskA_meter.avg),
                target_predict_tokens_avg=float(wandb_maskB_meter.avg),
                iter_time_ms=float(wandb_time_meter.avg),
                pass_time_s=float(window_seconds),
                images_seen=int(images_seen),
                anchor_passes_seen=float(images_seen) / float(max(1, anchor_count)),
                images_per_sec=float(window_images_per_sec),
                iterations_per_sec=float(window_iters_per_sec),
                lr=float(wandb_window_last_lr),
                wd=float(wandb_window_last_wd),
                global_batch_size=int(global_batch_size),
            )
            train_log = {
                "pass_index": int(max(0, pass_index)),
                "images_seen": int(images_seen),
            }
            update_log_dict("train", train_results, train_log, step="images_seen")
            log_images_seen_dict(train_log, images_seen=int(images_seen))

        final_snapshot = build_snapshot(
            pass_id=max(0, pass_index),
            loss_avg=float("nan"),
        )
        save_snapshot(
            final_snapshot,
            images_tag=int(images_seen),
            write_latest=True,
        )

        if should_stop_training:
            logger.info(f"Training terminated early by robustness early stopping at images_seen={images_seen}")
        training_completed_successfully = True
    except BaseException as exc:
        fatal_training_exception = exc
        logger.exception(
            "Training crashed before successful completion: images_seen=%d steps_done=%d pass_index=%d total_steps=%d total_images_budget=%d",
            int(images_seen),
            int(steps_done),
            int(pass_index),
            int(total_steps),
            int(total_images_budget),
        )
        raise
    finally:
        if rank == 0 and tuning_runtime is not None:
            try:
                process_tuning_completions(images_seen_at_log=int(images_seen))
            except Exception:
                logger.exception("Failed to process async tuning completions during shutdown")
            tuning_runtime.shutdown(wait=False, timeout_s=2.0)
        if wandb_enabled:
            wandb_exit_code = resolve_wandb_exit_code_from_outcome(
                training_completed_successfully=training_completed_successfully,
                fatal_training_exception=fatal_training_exception,
            )
            try:
                finish_wandb(exit_code=wandb_exit_code)
            except Exception:
                logger.exception("Failed to finalize W&B run")
                if fatal_training_exception is None:
                    raise
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                logger.warning("WARNING! Failed to destroy distributed process group cleanly: %s", exc)


if __name__ == "__main__":
    raise SystemExit("Use ijepath.train.main entrypoint")
