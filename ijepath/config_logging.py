import logging
import os
from typing import Any, Mapping

from omegaconf import OmegaConf


def render_config_yaml(cfg: Mapping[str, Any]) -> str:
    omega_cfg = OmegaConf.create(dict(cfg))
    OmegaConf.resolve(omega_cfg)
    return OmegaConf.to_yaml(omega_cfg)


def write_config(cfg: Mapping[str, Any], output_dir: str, name: str = "params-ijepa.yaml") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    omega_cfg = OmegaConf.create(dict(cfg))
    with open(path, "w", encoding="utf-8") as f:
        OmegaConf.save(config=omega_cfg, f=f)
    return path


def log_and_write_config(
    cfg: Mapping[str, Any],
    output_dir: str,
    *,
    logger: logging.Logger | None = None,
    name: str = "params-ijepa.yaml",
) -> str:
    if logger is None:
        logger = logging.getLogger("ijepath")
    logger.info(render_config_yaml(cfg))
    return write_config(cfg, output_dir=output_dir, name=name)

