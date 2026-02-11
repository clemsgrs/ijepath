from collections.abc import Sequence
import re
from typing import Any

from omegaconf import OmegaConf

_EARLY_STOPPING_METRIC_MODES: dict[str, dict[str, str]] = {
    "pathorob": {
        "ri": "max",
        "clustering_score": "max",
        "apd_id": "min",
        "apd_ood": "min",
        "apd_avg": "min",
    }
}


def load_training_config(
    *,
    config_file: str | None = None,
    fname: str | None = None,
    default_config: str | None = None,
    profile_config: str | None = None,
    run_config: str | None = None,
    opts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load training config from either a single file or merged layered files.

    Supported modes:
    - Single-file mode: `config_file` (+ optional CLI dotlist overrides in `opts`)
    - Layered mode: `default_config + profile_config + run_config` (+ `opts`)
    """
    opts_list = list(opts or [])
    if config_file is None:
        config_file = fname

    if config_file is not None and any(x is not None for x in (default_config, profile_config, run_config)):
        raise ValueError("Use either config_file or layered config files, not both.")

    if config_file is not None:
        cfg = OmegaConf.load(config_file)
        if opts_list:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts_list))
    else:
        missing = [
            name
            for name, value in (
                ("default_config", default_config),
                ("profile_config", profile_config),
                ("run_config", run_config),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Layered config mode requires default_config, profile_config, and run_config. "
                f"Missing: {', '.join(missing)}"
            )
        profile_cfg = _inject_profile_training_aliases(OmegaConf.load(profile_config))  # type: ignore[arg-type]
        cfg = OmegaConf.merge(
            OmegaConf.load(default_config),  # type: ignore[arg-type]
            profile_cfg,
            OmegaConf.load(run_config),  # type: ignore[arg-type]
            OmegaConf.from_dotlist(opts_list),
        )

    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    _project_profile_keys_into_data_if_missing(resolved)
    _attach_config_sources(
        resolved,
        config_file=config_file,
        default_config=default_config,
        profile_config=profile_config,
        run_config=run_config,
        opts=opts_list,
    )
    _validate_training_config(resolved)
    return resolved  # type: ignore[return-value]


def _inject_profile_training_aliases(profile_cfg):
    """Mirror profile geometry keys into profile.data.* so they merge into training config."""
    data_cfg = profile_cfg.get("data")
    if data_cfg is None:
        profile_cfg["data"] = {}
        data_cfg = profile_cfg["data"]

    key_map = {
        "context_mpp": "context_mpp",
        "target_mpp": "target_mpp",
        "context_fov_um": "context_fov_um",
        "target_fov_um": "target_fov_um",
        "targets_per_context": "targets_per_context",
        "spacing_tolerance": "spacing_tolerance",
    }

    for top_level_key, data_key in key_map.items():
        if top_level_key not in profile_cfg:
            continue
        top_level_value = profile_cfg[top_level_key]
        if top_level_value is None:
            continue

        existing_value = data_cfg.get(data_key)
        if existing_value is None:
            data_cfg[data_key] = top_level_value
        elif existing_value != top_level_value:
            raise ValueError(
                f"Conflicting values within profile config for {top_level_key} and data.{data_key}: "
                f"{top_level_value} vs {existing_value}"
            )

    return profile_cfg


def _project_profile_keys_into_data_if_missing(cfg: dict[str, Any]) -> None:
    """Single-file compatibility: fill data.* from top-level profile keys when missing."""
    data = cfg.setdefault("data", {})
    key_map = {
        "context_mpp": "context_mpp",
        "target_mpp": "target_mpp",
        "context_fov_um": "context_fov_um",
        "target_fov_um": "target_fov_um",
        "targets_per_context": "targets_per_context",
        "spacing_tolerance": "spacing_tolerance",
    }

    for top_level_key, data_key in key_map.items():
        if top_level_key not in cfg:
            continue
        top_level_value = cfg[top_level_key]
        if top_level_value is None:
            continue

        if data.get(data_key) is None:
            data[data_key] = top_level_value


def _attach_config_sources(
    cfg: dict[str, Any],
    *,
    config_file: str | None,
    default_config: str | None,
    profile_config: str | None,
    run_config: str | None,
    opts: list[str],
) -> None:
    if config_file is not None:
        cfg["_config_sources"] = {
            "mode": "single",
            "config_file": config_file,
            "opts": opts,
        }
        return

    cfg["_config_sources"] = {
        "mode": "layered",
        "default_config": default_config,
        "profile_config": profile_config,
        "run_config": run_config,
        "opts": opts,
    }


def _validate_training_config(cfg: dict[str, Any]) -> None:
    if "batch_size" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.batch_size")
    if "slide_metadata_index_jsonl" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.slide_metadata_index_jsonl")
    if "anchor_catalog_csv" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.anchor_catalog_csv")
    if "samples_per_chunk" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.samples_per_chunk")
    if "samples_per_epoch" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.samples_per_epoch")
    if "epochs" in cfg.get("optimization", {}):
        raise ValueError("Unsupported config value: optimization.epochs")
    if "checkpoint_every_epochs" in cfg.get("logging", {}):
        raise ValueError("Unsupported config value: logging.checkpoint_every_epochs")
    if "checkpoint_every_images" in cfg.get("logging", {}):
        raise ValueError("Unsupported config value: logging.checkpoint_every_images")
    if "log_every_images" in cfg.get("wandb", {}):
        raise ValueError("Unsupported config value: wandb.log_every_images")
    if "schedule" in cfg.get("tuning", {}):
        raise ValueError("Unsupported config value: tuning.schedule")

    required_paths = (
        ("data", "slide_manifest_csv"),
        ("data", "slide_metadata_parquet"),
        ("data", "anchor_catalog_manifest"),
    )
    required_geometry = (
        ("data", "context_mpp"),
        ("data", "target_mpp"),
        ("data", "context_fov_um"),
        ("data", "target_fov_um"),
        ("data", "targets_per_context"),
    )

    for section, key in required_paths + required_geometry:
        value = cfg.get(section, {}).get(key)
        if value is None:
            raise ValueError(f"Missing required config value: {section}.{key}")

    architecture = cfg.get("meta", {}).get("architecture")
    if architecture is None:
        raise ValueError("Missing required config value: meta.architecture")
    patch_size = cfg.get("meta", {}).get("patch_size")
    if patch_size is None:
        raise ValueError("Missing required config value: meta.patch_size")
    if int(patch_size) <= 0:
        raise ValueError("meta.patch_size must be > 0")

    batch_size_per_gpu = cfg.get("data", {}).get("batch_size_per_gpu")
    if batch_size_per_gpu is None:
        raise ValueError("Missing required config value: data.batch_size_per_gpu")
    if int(batch_size_per_gpu) <= 0:
        raise ValueError("data.batch_size_per_gpu must be > 0")

    total_images_budget = cfg.get("optimization", {}).get("total_images_budget")
    if total_images_budget is None:
        raise ValueError("Missing required config value: optimization.total_images_budget")
    if int(total_images_budget) <= 0:
        raise ValueError("optimization.total_images_budget must be > 0")

    training_log_every = cfg.get("training", {}).get("log_every", None)
    if training_log_every is not None and int(training_log_every) <= 0:
        raise ValueError("training.log_every must be > 0")
    training_save_every = cfg.get("training", {}).get("save_every", None)
    if training_save_every is not None and int(training_save_every) <= 0:
        raise ValueError("training.save_every must be > 0")

    tune_every = cfg.get("tuning", {}).get("tune_every", None)
    if tune_every is not None and int(tune_every) <= 0:
        raise ValueError("tuning.tune_every must be > 0")

    low_anchor_pass_warning_threshold = float(
        cfg.get("data", {}).get("low_anchor_pass_warning_threshold", 1.0)
    )
    high_anchor_pass_warning_threshold = float(
        cfg.get("data", {}).get("high_anchor_pass_warning_threshold", 5.0)
    )
    if low_anchor_pass_warning_threshold <= 0:
        raise ValueError("data.low_anchor_pass_warning_threshold must be > 0")
    if high_anchor_pass_warning_threshold <= 0:
        raise ValueError("data.high_anchor_pass_warning_threshold must be > 0")
    if high_anchor_pass_warning_threshold <= low_anchor_pass_warning_threshold:
        raise ValueError(
            "data.high_anchor_pass_warning_threshold must be > "
            "data.low_anchor_pass_warning_threshold"
        )

    targets_per_context = int(cfg["data"]["targets_per_context"])
    mask_cfg = cfg.setdefault("mask", {})
    num_pred_masks = mask_cfg.get("num_pred_masks")
    if num_pred_masks is None:
        mask_cfg["num_pred_masks"] = targets_per_context
    elif int(num_pred_masks) != targets_per_context:
        raise ValueError(
            "mask.num_pred_masks must match data.targets_per_context "
            f"(got {num_pred_masks} vs {targets_per_context})"
        )

    tuning_cfg = dict(cfg.get("tuning", {}) or {})
    if not bool(tuning_cfg.get("enable", False)):
        return

    execution_cfg = dict(tuning_cfg.get("execution", {}) or {})
    execution_mode = str(execution_cfg.get("mode", "")).strip().lower()
    if execution_mode != "async":
        raise ValueError("tuning.execution.mode must be 'async' when tuning.enable=true")

    execution_device = str(execution_cfg.get("device", "")).strip().lower()
    if execution_device != "auto" and re.fullmatch(r"cuda:\d+", execution_device) is None:
        raise ValueError("tuning.execution.device must be 'auto' or match 'cuda:<id>' when tuning.enable=true")

    max_pending_jobs = int(execution_cfg.get("max_pending_jobs", 2))
    if max_pending_jobs <= 0:
        raise ValueError("tuning.execution.max_pending_jobs must be > 0")

    coalesce_policy = str(execution_cfg.get("coalesce_policy", "newest")).strip().lower()
    if coalesce_policy != "newest":
        raise ValueError("tuning.execution.coalesce_policy must be 'newest'")

    poll_every_steps = int(execution_cfg.get("poll_every_steps", 10))
    if poll_every_steps <= 0:
        raise ValueError("tuning.execution.poll_every_steps must be > 0")

    keep_last_n_snapshots = int(execution_cfg.get("keep_last_n_snapshots", 2))
    if keep_last_n_snapshots < 0:
        raise ValueError("tuning.execution.keep_last_n_snapshots must be >= 0")

    plugins = list(tuning_cfg.get("plugins", []) or [])
    enabled_plugins = [dict(p) for p in plugins if bool(dict(p).get("enable", True))]

    valid_plugin_types = {"pathorob"}
    for plugin in enabled_plugins:
        ptype = str(plugin.get("type", "")).strip().lower()
        if ptype not in valid_plugin_types:
            raise ValueError(f"Unknown tuning plugin type: {ptype}")
        legacy_early_stopping_keys = [
            k for k in ("use_for_early_stopping", "early_stopping_metric", "early_stopping_mode") if k in plugin
        ]
        if legacy_early_stopping_keys:
            raise ValueError(
                "Unsupported plugin-level early stopping keys in tuning.plugins entry: "
                f"{legacy_early_stopping_keys}. "
                "Use tuning.early_stopping.selection.{plugin,dataset,metric} instead."
            )
        if ptype == "pathorob":
            feature_num_workers = int(plugin.get("feature_num_workers", plugin.get("num_workers", 4)))
            if feature_num_workers < 0:
                raise ValueError("tuning.plugins.pathorob.feature_num_workers must be >= 0")
            feature_prefetch_factor = int(plugin.get("feature_prefetch_factor", 4))
            if feature_prefetch_factor <= 0:
                raise ValueError("tuning.plugins.pathorob.feature_prefetch_factor must be > 0")

            metric_default_every = {
                "ri": 1,
                "apd": 5,
                "clustering": 5,
            }
            for metric_section, default_every in metric_default_every.items():
                metric_cfg = dict(plugin.get(metric_section, {}) or {})
                every_n = int(metric_cfg.get("every_n_evals", default_every))
                if every_n <= 0:
                    raise ValueError(
                        f"tuning.plugins.pathorob.{metric_section}.every_n_evals must be > 0"
                    )

            apd_cfg = dict(plugin.get("apd", {}) or {})
            mode = str(apd_cfg.get("mode", "paper"))
            if mode == "paper":
                datasets_cfg = dict(plugin.get("datasets", {}) or {})
                camelyon_cfg = dict(datasets_cfg.get("camelyon", {}) or {})
                if bool(camelyon_cfg.get("enable", False)):
                    id_centers = sorted(list(camelyon_cfg.get("id_centers", [])))
                    ood_centers = sorted(list(camelyon_cfg.get("ood_centers", [])))
                    if id_centers != ["RUMC", "UMCU"]:
                        raise ValueError(
                            "PathoROB paper mode requires camelyon id_centers=['RUMC', 'UMCU']"
                        )
                    if ood_centers != ["CWZ", "LPON", "RST"]:
                        raise ValueError(
                            "PathoROB paper mode requires camelyon ood_centers=['CWZ', 'RST', 'LPON']"
                        )

    early_cfg = dict(tuning_cfg.get("early_stopping", {}) or {})
    if not bool(early_cfg.get("enable", False)):
        return

    selection_cfg = dict(early_cfg.get("selection", {}) or {})
    plugin_name = str(selection_cfg.get("plugin", "")).strip().lower()
    dataset_name = str(selection_cfg.get("dataset", "")).strip()
    metric_name = str(selection_cfg.get("metric", "")).strip()

    if not plugin_name or not dataset_name or not metric_name:
        raise ValueError(
            "tuning.early_stopping.enable=true requires tuning.early_stopping.selection "
            "with non-empty plugin, dataset, and metric"
        )

    matching_plugins = [p for p in enabled_plugins if str(p.get("type", "")).strip().lower() == plugin_name]
    if len(matching_plugins) == 0:
        raise ValueError(
            f"tuning.early_stopping.selection.plugin='{plugin_name}' must match an enabled plugin type"
        )
    if len(matching_plugins) > 1:
        raise ValueError(
            f"tuning.early_stopping.selection.plugin='{plugin_name}' is ambiguous: "
            "multiple enabled plugin entries share this type"
        )

    selected_plugin = dict(matching_plugins[0])
    if plugin_name == "pathorob":
        datasets_cfg = dict(selected_plugin.get("datasets", {}) or {})
        dataset_cfg = dict(datasets_cfg.get(dataset_name, {}) or {})
        if not bool(dataset_cfg.get("enable", False)):
            raise ValueError(
                f"tuning.early_stopping.selection.dataset='{dataset_name}' must reference an enabled "
                f"{plugin_name} dataset entry"
            )
        metric_to_section = {
            "ri": "ri",
            "apd_id": "apd",
            "apd_ood": "apd",
            "apd_avg": "apd",
            "clustering_score": "clustering",
        }
        selected_section = metric_to_section.get(metric_name)
        if selected_section is not None:
            section_cfg = dict(selected_plugin.get(selected_section, {}) or {})
            default_every = 1 if selected_section == "ri" else 5
            section_every_n = int(section_cfg.get("every_n_evals", default_every))
            if section_every_n != 1:
                raise ValueError(
                    "tuning.early_stopping.selection.metric cadence must run every eval "
                    f"(set tuning.plugins.pathorob.{selected_section}.every_n_evals=1)"
                )

    metric_modes = _EARLY_STOPPING_METRIC_MODES.get(plugin_name, {})
    if metric_name not in metric_modes:
        raise ValueError(
            f"Unsupported tuning.early_stopping.selection.metric='{metric_name}' for plugin='{plugin_name}'. "
            f"Allowed metrics: {sorted(metric_modes.keys())}"
        )
