from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf


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
    _normalize_batch_size_keys(resolved)
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


def _normalize_batch_size_keys(cfg: dict[str, Any]) -> None:
    data = cfg.setdefault("data", {})
    per_gpu = data.get("batch_size_per_gpu")
    legacy = data.get("batch_size")

    if per_gpu is None and legacy is None:
        return
    if per_gpu is None:
        data["batch_size_per_gpu"] = int(legacy)
        return
    if legacy is None:
        return
    if int(per_gpu) != int(legacy):
        raise ValueError(
            "Conflicting values for data.batch_size_per_gpu and data.batch_size: "
            f"{per_gpu} vs {legacy}"
        )


def _validate_training_config(cfg: dict[str, Any]) -> None:
    if "samples_per_chunk" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.samples_per_chunk")
    if "samples_per_epoch" in cfg.get("data", {}):
        raise ValueError("Unsupported config value: data.samples_per_epoch")
    if "epochs" in cfg.get("optimization", {}):
        raise ValueError("Unsupported config value: optimization.epochs")
    if "checkpoint_every_epochs" in cfg.get("logging", {}):
        raise ValueError("Unsupported config value: logging.checkpoint_every_epochs")

    required_paths = (
        ("data", "slide_manifest_csv"),
        ("data", "slide_metadata_index_jsonl"),
        ("data", "anchor_catalog_csv"),
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

    plugins = list(tuning_cfg.get("plugins", []) or [])
    enabled_plugins = [dict(p) for p in plugins if bool(dict(p).get("enable", True))]

    valid_plugin_types = {"pathorob"}
    for plugin in enabled_plugins:
        ptype = str(plugin.get("type", "")).strip().lower()
        if ptype not in valid_plugin_types:
            raise ValueError(f"Unknown tuning plugin type: {ptype}")
        if ptype == "pathorob":
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

    selected = [p for p in enabled_plugins if bool(p.get("use_for_early_stopping", False))]
    if len(selected) == 0:
        raise ValueError(
            "tuning.early_stopping.enable=true requires one enabled plugin with use_for_early_stopping=true"
        )
    if len(selected) > 1:
        raise ValueError("At most one enabled plugin may set use_for_early_stopping=true")

    selected_plugin = dict(selected[0])
    metric_key = str(selected_plugin.get("early_stopping_metric", "")).strip()
    if not metric_key:
        raise ValueError(
            "Selected early-stopping plugin must define a non-empty early_stopping_metric"
        )

    mode = str(selected_plugin.get("early_stopping_mode", "max")).strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError(
            f"Selected early-stopping plugin has invalid early_stopping_mode: {mode}"
        )
