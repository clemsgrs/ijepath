import io
import logging
from pathlib import Path

import yaml

from ijepath.config_logging import log_and_write_config


def test_log_and_write_config_emits_yaml_and_persists_file(tmp_path: Path):
    cfg = {
        "data": {"batch_size_per_gpu": 2},
        "logging": {"folder": str(tmp_path), "write_tag": "unit-test"},
        "_config_sources": {"mode": "layered"},
    }

    logger = logging.getLogger("ijepath.test.config_logging")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.propagate = False

    out_path = log_and_write_config(cfg, output_dir=str(tmp_path), logger=logger)

    assert Path(out_path).exists()
    saved = yaml.safe_load(Path(out_path).read_text(encoding="utf-8"))
    assert saved["data"]["batch_size_per_gpu"] == 2
    assert saved["_config_sources"]["mode"] == "layered"

    logged = stream.getvalue()
    assert "batch_size_per_gpu: 2" in logged
    assert "_config_sources:" in logged

