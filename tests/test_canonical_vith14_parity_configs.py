from pathlib import Path

import pytest

from ijepath.config_loading import load_training_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_canonical_vith14_parity_smoke_config_resolves_expected_values():
    repo_root = _repo_root()
    cfg = load_training_config(
        default_config=str(repo_root / "configs/defaults_canonical.yaml"),
        profile_config=str(
            repo_root / "configs/profiles/canonical_vith14_parity_20x_256_224.yaml"
        ),
        run_config=str(repo_root / "configs/runs/canonical_vith14_parity_smoke.yaml"),
        opts=[
            "data.slide_manifest_csv=/tmp/slides.csv",
            "data.slide_metadata_parquet=/tmp/slide_metadata.parquet",
            "data.anchor_catalog_manifest=/tmp/anchor_catalog_manifest.json",
        ],
    )

    assert cfg["pretraining"]["mode"] == "canonical"
    assert cfg["meta"]["architecture"] == "vit_huge"
    assert int(cfg["meta"]["patch_size"]) == 14
    assert int(cfg["meta"]["pred_depth"]) == 12
    assert int(cfg["meta"]["pred_emb_dim"]) == 384
    assert cfg["canonical"]["crop_scale"] == [0.85, 1.0]
    assert cfg["canonical"]["use_horizontal_flip"] is False
    assert cfg["canonical"]["use_color_distortion"] is False
    assert cfg["canonical"]["use_gaussian_blur"] is False
    assert cfg["canonical"]["allow_overlap"] is False
    assert int(cfg["canonical"]["min_keep"]) == 10
    assert int(cfg["canonical"]["num_pred_masks"]) == 4
    assert float(cfg["optimization"]["warmup"]) == pytest.approx(40.0 / 300.0)
    assert "log_freq_steps" not in cfg["logging"]


def test_canonical_vith14_parity_run_uses_absolute_budget_and_warmup_ratio():
    repo_root = _repo_root()
    cfg = load_training_config(
        default_config=str(repo_root / "configs/defaults_canonical.yaml"),
        profile_config=str(
            repo_root / "configs/profiles/canonical_vith14_parity_20x_256_224.yaml"
        ),
        run_config=str(
            repo_root / "configs/runs/canonical_vith14_parity_pathorob_camelyon.yaml"
        ),
        opts=[
            "data.slide_manifest_csv=/tmp/slides.csv",
            "data.slide_metadata_parquet=/tmp/slide_metadata.parquet",
            "data.anchor_catalog_manifest=/tmp/anchor_catalog_manifest.json",
        ],
    )

    assert int(cfg["optimization"]["total_images_budget"]) == 384_350_100
    assert float(cfg["optimization"]["warmup"]) == pytest.approx(40.0 / 300.0)
    assert "epochs_equivalent" not in cfg["optimization"]
    assert "warmup_epochs" not in cfg["optimization"]
