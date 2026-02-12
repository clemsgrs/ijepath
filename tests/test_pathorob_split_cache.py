from pathlib import Path

import pandas as pd
import torch

from ijepath.eval.plugins.pathorob import PathoROBPlugin


def _build_manifest_csv(path: Path) -> Path:
    rows = []
    sample_idx = 0
    for label in ["normal", "tumor"]:
        for center in ["RUMC", "UMCU", "CWZ"]:
            for slide in range(5):
                slide_id = f"{center}_{label}_{slide}"
                for _ in range(2):
                    rows.append(
                        {
                            "sample_id": f"s{sample_idx}",
                            "image_path": f"/tmp/fake/{sample_idx}.png",
                            "label": label,
                            "medical_center": center,
                            "slide_id": slide_id,
                        }
                    )
                    sample_idx += 1
    df = pd.DataFrame(rows)
    csv_path = path / "manifest.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _plugin_cfg(manifest_csv: Path, shared_cache_root: Path, repetitions: int = 1) -> dict:
    return {
        "type": "pathorob",
        "enable": True,
        "shared_cache_root": str(shared_cache_root),
        "datasets": {
            "camelyon": {
                "enable": True,
                "manifest_csv": str(manifest_csv),
                "id_centers": ["RUMC", "UMCU"],
                "ood_centers": ["CWZ"],
            }
        },
        "apd": {
            "enable": True,
            "mode": "custom",
            "correlation_levels": [0.0, 1.0],
            "repetitions": int(repetitions),
            "id_test_fraction": 0.2,
        },
    }


def test_pathorob_split_cache_reuses_shared_splits(tmp_path: Path, monkeypatch):
    manifest_csv = _build_manifest_csv(tmp_path)
    shared_cache_root = tmp_path / "cache"

    plugin1 = PathoROBPlugin(
        cfg=_plugin_cfg(manifest_csv=manifest_csv, shared_cache_root=shared_cache_root, repetitions=1),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run1" / "tuning",
    )
    dataset_cfg = dict(plugin1.cfg["datasets"]["camelyon"])
    manifest_df = plugin1._load_manifest("camelyon", str(manifest_csv))
    splits1 = plugin1._ensure_apd_splits("camelyon", dataset_cfg, manifest_df)
    assert len(splits1) == 2

    cache_root = shared_cache_root / "pathorob" / "apd_splits"
    cache_dirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
    assert len(cache_dirs) == 1

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("generate_apd_splits should not run on cache hit")

    monkeypatch.setattr("ijepath.eval.plugins.pathorob.generate_apd_splits", _raise_if_called)

    plugin2 = PathoROBPlugin(
        cfg=_plugin_cfg(manifest_csv=manifest_csv, shared_cache_root=shared_cache_root, repetitions=1),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run2" / "tuning",
    )
    dataset_cfg2 = dict(plugin2.cfg["datasets"]["camelyon"])
    manifest_df2 = plugin2._load_manifest("camelyon", str(manifest_csv))
    splits2 = plugin2._ensure_apd_splits("camelyon", dataset_cfg2, manifest_df2)
    assert len(splits2) == 2
    assert (tmp_path / "run2" / "tuning" / "pathorob" / "splits" / "camelyon" / "rep_00" / "split_01.csv").is_file()


def test_pathorob_split_cache_key_changes_when_params_change(tmp_path: Path):
    manifest_csv = _build_manifest_csv(tmp_path)
    shared_cache_root = tmp_path / "cache"

    plugin1 = PathoROBPlugin(
        cfg=_plugin_cfg(manifest_csv=manifest_csv, shared_cache_root=shared_cache_root, repetitions=1),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run1" / "tuning",
    )
    dataset_cfg1 = dict(plugin1.cfg["datasets"]["camelyon"])
    manifest_df1 = plugin1._load_manifest("camelyon", str(manifest_csv))
    plugin1._ensure_apd_splits("camelyon", dataset_cfg1, manifest_df1)

    cache_root = shared_cache_root / "pathorob" / "apd_splits"
    cache_keys_before = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    assert len(cache_keys_before) == 1

    plugin2 = PathoROBPlugin(
        cfg=_plugin_cfg(manifest_csv=manifest_csv, shared_cache_root=shared_cache_root, repetitions=2),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run2" / "tuning",
    )
    dataset_cfg2 = dict(plugin2.cfg["datasets"]["camelyon"])
    manifest_df2 = plugin2._load_manifest("camelyon", str(manifest_csv))
    plugin2._ensure_apd_splits("camelyon", dataset_cfg2, manifest_df2)

    cache_keys_after = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    assert len(cache_keys_after) == 2
    assert set(cache_keys_before).issubset(set(cache_keys_after))
