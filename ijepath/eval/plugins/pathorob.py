from __future__ import annotations

import hashlib
import json
import logging
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from ijepath.eval.dataset import EvalDataset
from ijepath.eval.features import extract_teacher_features_single_process
from ijepath.eval.pathorob.allocations import get_paper_metadata
from ijepath.eval.pathorob.apd import compute_apd
from ijepath.eval.pathorob.clustering import compute_clustering_score
from ijepath.eval.pathorob.datasets import load_manifest
from ijepath.eval.pathorob.ri import compute_ri
from ijepath.eval.pathorob.splits import generate_apd_splits

from .base import BenchmarkPlugin, PluginResult

logger = logging.getLogger("ijepath")


class PathoROBPlugin(BenchmarkPlugin):
    """PathoROB robustness tuning plugin (teacher-only)."""

    def __init__(self, cfg: dict, device: torch.device, output_dir: Path):
        self.cfg = dict(cfg or {})
        self.device = device
        self.name = "pathorob"

        self.output_dir = Path(output_dir)
        self.base_dir = self.output_dir / "pathorob"
        self.splits_dir = self.base_dir / "splits"
        self.metrics_dir = self.base_dir / "metrics"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        shared_cache_root_raw = str(self.cfg.get("shared_cache_root", "") or "").strip()
        self.shared_split_cache_dir: Path | None = None
        if shared_cache_root_raw:
            self.shared_split_cache_dir = (
                Path(shared_cache_root_raw).expanduser().resolve() / "pathorob" / "apd_splits"
            )
            self.shared_split_cache_dir.mkdir(parents=True, exist_ok=True)

        transform_cfg = dict(self.cfg.get("transforms", {}) or {})
        resize = int(transform_cfg.get("resize", 256))
        crop_size_cfg = transform_cfg.get("crop_size", None)
        crop_size = None if crop_size_cfg is None else int(crop_size_cfg)
        normalize = transform_cfg.get("normalize", "imagenet")
        if normalize == "imagenet":
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            raise ValueError(f"Unsupported normalization preset: {normalize}")

        if resize <= 0:
            raise ValueError("PathoROB transform resize must be > 0")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("PathoROB transform crop_size must be > 0 when provided")

        transform_steps = [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        if crop_size is not None:
            transform_steps.append(transforms.CenterCrop(crop_size))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

        self._manifest_cache: dict[str, pd.DataFrame] = {}
        self._apd_split_cache: dict[str, list[pd.DataFrame]] = {}
        self._dataset_cache: dict[str, dict[str, object]] = {}

    def should_run(self, images_seen: int, tune_index: int) -> bool:
        tune_every_images = self.cfg.get("tune_every_images", None)
        if tune_every_images in (None, 0):
            return True
        interval = int(tune_every_images)
        if interval <= 0:
            return True
        return int(images_seen) % interval == 0

    def _load_manifest(self, dataset_name: str, csv_path: str) -> pd.DataFrame:
        if dataset_name in self._manifest_cache:
            return self._manifest_cache[dataset_name]
        df = load_manifest(csv_path, dataset_name)
        self._manifest_cache[dataset_name] = df
        return df

    def _extract_features(self, teacher_backbone, manifest_df: pd.DataFrame) -> np.ndarray:
        feature_num_workers = int(self.cfg.get("feature_num_workers", self.cfg.get("num_workers", 4)))
        feature_persistent_workers = bool(self.cfg.get("feature_persistent_workers", True))
        feature_prefetch_factor = int(self.cfg.get("feature_prefetch_factor", 4))
        dataset = EvalDataset(
            manifest_df,
            transform=self.transform,
            image_col="image_path",
            label_col="label",
        )
        loader_kwargs: dict[str, object] = {}
        if feature_num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(feature_persistent_workers)
            loader_kwargs["prefetch_factor"] = int(feature_prefetch_factor)
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=int(self.cfg.get("batch_size_per_gpu", 32)),
            num_workers=feature_num_workers,
            pin_memory=True,
            drop_last=False,
            **loader_kwargs,
        )
        return extract_teacher_features_single_process(
            teacher=teacher_backbone,
            loader=loader,
            device=self.device,
        )

    def _dataset_context(self, dataset_name: str, manifest_df: pd.DataFrame) -> dict[str, object]:
        cached = self._dataset_cache.get(dataset_name)
        if cached is not None:
            return cached

        ctx = {
            "sample_to_idx": {sid: i for i, sid in enumerate(manifest_df["sample_id"].tolist())},
            "label_codes": pd.factorize(manifest_df["label"])[0].astype(int),
            "center_codes": pd.factorize(manifest_df["medical_center"])[0].astype(int),
            "slide_ids": manifest_df["slide_id"].astype(str).to_numpy(),
        }
        self._dataset_cache[dataset_name] = ctx
        return ctx

    @staticmethod
    def _metric_enabled(
        metric_cfg: dict,
        *,
        tune_index: int,
        default_every_n: int,
    ) -> bool:
        if not bool(metric_cfg.get("enable", True)):
            return False
        every_n = int(metric_cfg.get("every_n_evals", default_every_n))
        if every_n <= 0:
            return False
        return int(tune_index) % int(every_n) == 0

    def _get_split_params(self, dataset_cfg: dict) -> dict:
        apd_cfg = dict(self.cfg.get("apd", {}) or {})
        mode = str(apd_cfg.get("mode", "paper"))
        return {
            "mode": mode,
            "repetitions": int(apd_cfg.get("repetitions", 3)),
            "id_test_fraction": float(apd_cfg.get("id_test_fraction", 0.2)),
            "seed": int(self.cfg.get("seed", 0)),
            "id_centers": sorted(list(dataset_cfg.get("id_centers", []))),
            "ood_centers": sorted(list(dataset_cfg.get("ood_centers", []))),
            "correlation_levels": sorted(list(apd_cfg.get("correlation_levels", []))) if mode == "custom" else None,
        }

    def _validate_paper_mode_centers(self, dataset_name: str, dataset_cfg: dict) -> None:
        mode = str(dict(self.cfg.get("apd", {}) or {}).get("mode", "paper"))
        if mode != "paper":
            return

        try:
            _paper_classes, paper_id, paper_ood = get_paper_metadata(dataset_name)
        except ValueError:
            return

        config_id = sorted(list(dataset_cfg.get("id_centers", [])))
        config_ood = sorted(list(dataset_cfg.get("ood_centers", [])))

        if config_id != sorted(paper_id):
            raise ValueError(
                f"[APD] Paper mode requires id_centers={paper_id} for {dataset_name}, "
                f"but config has {config_id}."
            )
        if config_ood != sorted(paper_ood):
            raise ValueError(
                f"[APD] Paper mode requires ood_centers={paper_ood} for {dataset_name}, "
                f"but config has {config_ood}."
            )

    @staticmethod
    def _fingerprint_manifest_csv(csv_path: str) -> str:
        path = Path(csv_path).expanduser().resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _split_cache_key(*, dataset_name: str, split_params: dict, manifest_fingerprint: str) -> str:
        payload = {
            "dataset_name": str(dataset_name),
            "split_params": dict(split_params),
            "manifest_fingerprint": str(manifest_fingerprint),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_split_frames(split_files: list[Path]) -> list[pd.DataFrame]:
        return [pd.read_csv(path) for path in split_files]

    def _ensure_apd_splits(self, dataset_name: str, dataset_cfg: dict, manifest_df: pd.DataFrame) -> list[pd.DataFrame]:
        if dataset_name in self._apd_split_cache:
            return self._apd_split_cache[dataset_name]

        self._validate_paper_mode_centers(dataset_name, dataset_cfg)

        apd_cfg = dict(self.cfg.get("apd", {}) or {})
        split_dir = self.splits_dir / dataset_name
        metadata_file = split_dir / "split_params.json"
        split_files = sorted(split_dir.glob("rep_*/split_*.csv"))

        current_params = self._get_split_params(dataset_cfg)
        manifest_csv = str(dataset_cfg.get("manifest_csv", ""))
        manifest_fingerprint = self._fingerprint_manifest_csv(manifest_csv)
        current_params_with_manifest = dict(current_params)
        current_params_with_manifest["manifest_fingerprint"] = str(manifest_fingerprint)
        need_regen = True

        if split_files and metadata_file.exists():
            try:
                saved = json.loads(metadata_file.read_text(encoding="utf-8"))
                if saved == current_params_with_manifest:
                    need_regen = False
                    logger.info("[APD] Loading cached splits for %s", dataset_name)
            except Exception:
                need_regen = True

        cache_dataset_dir: Path | None = None
        if self.shared_split_cache_dir is not None:
            cache_key = self._split_cache_key(
                dataset_name=dataset_name,
                split_params=current_params,
                manifest_fingerprint=manifest_fingerprint,
            )
            cache_dataset_dir = self.shared_split_cache_dir / cache_key / str(dataset_name)
            cache_metadata_file = self.shared_split_cache_dir / cache_key / "split_params.json"
            cache_split_files = sorted(cache_dataset_dir.glob("rep_*/split_*.csv"))
            if need_regen and cache_split_files and cache_metadata_file.exists():
                try:
                    saved_cache_params = json.loads(cache_metadata_file.read_text(encoding="utf-8"))
                    if saved_cache_params == current_params_with_manifest:
                        split_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(cache_dataset_dir, split_dir, dirs_exist_ok=True)
                        metadata_file.write_text(
                            json.dumps(current_params_with_manifest, indent=2),
                            encoding="utf-8",
                        )
                        split_files = sorted(split_dir.glob("rep_*/split_*.csv"))
                        if split_files:
                            need_regen = False
                            logger.info("[APD] Reused shared split cache for %s (key=%s)", dataset_name, cache_key)
                except Exception:
                    need_regen = True

        if not need_regen:
            splits = self._load_split_frames(split_files)
        else:
            splits = generate_apd_splits(
                df=manifest_df,
                output_dir=self.splits_dir,
                dataset_name=dataset_name,
                repetitions=int(apd_cfg.get("repetitions", 3)),
                correlation_levels=list(apd_cfg.get("correlation_levels", [])),
                id_centers=list(dataset_cfg.get("id_centers", [])),
                ood_centers=list(dataset_cfg.get("ood_centers", [])),
                id_test_fraction=float(apd_cfg.get("id_test_fraction", 0.2)),
                seed=int(self.cfg.get("seed", 0)),
                mode=str(apd_cfg.get("mode", "paper")),
            )
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(json.dumps(current_params_with_manifest, indent=2), encoding="utf-8")
            if cache_dataset_dir is not None:
                cache_key_dir = cache_dataset_dir.parent
                cache_key_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(split_dir, cache_dataset_dir, dirs_exist_ok=True)
                (cache_key_dir / "split_params.json").write_text(
                    json.dumps(current_params_with_manifest, indent=2),
                    encoding="utf-8",
                )

        self._apd_split_cache[dataset_name] = splits
        return splits

    def _run_dataset(
        self,
        dataset_name: str,
        dataset_cfg: dict,
        teacher_backbone,
        tune_index: int,
        images_seen: int,
    ) -> tuple[list[dict], dict[str, float]]:
        manifest_df = self._load_manifest(dataset_name, str(dataset_cfg["manifest_csv"]))
        features = self._extract_features(teacher_backbone=teacher_backbone, manifest_df=manifest_df)
        dataset_ctx = self._dataset_context(dataset_name, manifest_df)
        sample_to_idx = dict(dataset_ctx["sample_to_idx"])

        metric_rows: list[dict] = []
        log_metrics: dict[str, float] = {}

        ri_cfg = dict(self.cfg.get("ri", {}) or {})
        apd_cfg = dict(self.cfg.get("apd", {}) or {})
        cl_cfg = dict(self.cfg.get("clustering", {}) or {})

        if self._metric_enabled(ri_cfg, tune_index=int(tune_index), default_every_n=1):
            ri = compute_ri(
                dataset_name=dataset_name,
                features=features,
                manifest_df=manifest_df,
                k_candidates=list(ri_cfg.get("k_candidates", [3, 5, 7, 10, 15])),
                max_pairs=(None if int(self.cfg.get("max_pairs", 0)) <= 0 else int(self.cfg.get("max_pairs", 0))),
                random_state=int(self.cfg.get("seed", 0)) + int(tune_index),
            )
            metric_rows.append(
                {
                    "plugin": self.name,
                    "dataset": dataset_name,
                    "metric": "ri",
                    "tune_index": int(tune_index),
                    "images_seen": int(images_seen),
                    "value": float(ri.value),
                    "std": float(ri.std),
                    "n": int(ri.n_pairs),
                    "extra": json.dumps({"k": int(ri.k)}),
                }
            )
            log_metrics[f"{dataset_name}/ri"] = float(ri.value)

        if self._metric_enabled(apd_cfg, tune_index=int(tune_index), default_every_n=5):
            split_frames = self._ensure_apd_splits(dataset_name, dataset_cfg, manifest_df)
            aligned: list[pd.DataFrame] = []
            for sp in split_frames:
                s = sp.copy()
                s["feature_index"] = s["sample_id"].map(sample_to_idx)
                s = s[s["feature_index"].notna()].copy()
                s["feature_index"] = s["feature_index"].astype(int)
                aligned.append(s)

            apd = compute_apd(
                dataset_name=dataset_name,
                features=features,
                all_splits=aligned,
                seed=int(self.cfg.get("seed", 0)) + int(tune_index),
            )

            metric_rows.extend(
                [
                    {
                        "plugin": self.name,
                        "dataset": dataset_name,
                        "metric": "apd_id",
                        "tune_index": int(tune_index),
                        "images_seen": int(images_seen),
                        "value": float(apd.apd_id),
                        "std": float(apd.apd_id_std),
                        "n": int(apd_cfg.get("repetitions", 3)),
                        "extra": "{}",
                    },
                    {
                        "plugin": self.name,
                        "dataset": dataset_name,
                        "metric": "apd_ood",
                        "tune_index": int(tune_index),
                        "images_seen": int(images_seen),
                        "value": float(apd.apd_ood),
                        "std": float(apd.apd_ood_std),
                        "n": int(apd_cfg.get("repetitions", 3)),
                        "extra": "{}",
                    },
                    {
                        "plugin": self.name,
                        "dataset": dataset_name,
                        "metric": "apd_avg",
                        "tune_index": int(tune_index),
                        "images_seen": int(images_seen),
                        "value": float(apd.apd_avg),
                        "std": float(apd.apd_avg_std),
                        "n": int(apd_cfg.get("repetitions", 3)),
                        "extra": "{}",
                    },
                ]
            )
            log_metrics[f"{dataset_name}/apd_id"] = float(apd.apd_id)
            log_metrics[f"{dataset_name}/apd_ood"] = float(apd.apd_ood)
            log_metrics[f"{dataset_name}/apd_avg"] = float(apd.apd_avg)

            for rho, (mean_acc, _std) in apd.acc_id_by_rho.items():
                rho_str = f"{rho:.2f}".replace(".", "_")
                log_metrics[f"{dataset_name}/acc_id_rho{rho_str}"] = float(mean_acc)
            for rho, (mean_acc, _std) in apd.acc_ood_by_rho.items():
                rho_str = f"{rho:.2f}".replace(".", "_")
                log_metrics[f"{dataset_name}/acc_ood_rho{rho_str}"] = float(mean_acc)

        if self._metric_enabled(cl_cfg, tune_index=int(tune_index), default_every_n=5):
            cl = compute_clustering_score(
                dataset_name=dataset_name,
                features=features,
                manifest_df=manifest_df,
                repeats=int(cl_cfg.get("repeats", 5)),
                k_min=int(cl_cfg.get("k_min", 2)),
                k_max=int(cl_cfg.get("k_max", 10)),
                max_pairs=(None if int(self.cfg.get("max_pairs", 0)) <= 0 else int(self.cfg.get("max_pairs", 0))),
                random_state=int(self.cfg.get("seed", 0)) + int(tune_index),
            )
            metric_rows.append(
                {
                    "plugin": self.name,
                    "dataset": dataset_name,
                    "metric": "clustering_score",
                    "tune_index": int(tune_index),
                    "images_seen": int(images_seen),
                    "value": float(cl.score),
                    "std": float(cl.std),
                    "n": int(cl.n_pairs),
                    "extra": "{}",
                }
            )
            log_metrics[f"{dataset_name}/clustering_score"] = float(cl.score)

        return metric_rows, log_metrics

    @torch.no_grad()
    def run(self, teacher, tune_index: int, images_seen: int) -> PluginResult:
        teacher_backbone = teacher.module if hasattr(teacher, "module") else teacher
        teacher_backbone.eval()

        all_rows: list[dict] = []
        all_logs: dict[str, float] = {}

        datasets_cfg = dict(self.cfg.get("datasets", {}) or {})
        any_enabled = any(bool(dict(v).get("enable", False)) for v in datasets_cfg.values())
        if not any_enabled:
            logger.warning("WARNING! [PathoROB] No datasets are enabled for tuning.")

        for dataset_name, raw_cfg in datasets_cfg.items():
            dataset_cfg = dict(raw_cfg or {})
            if not bool(dataset_cfg.get("enable", False)):
                continue
            try:
                rows, logs = self._run_dataset(
                    dataset_name=str(dataset_name),
                    dataset_cfg=dataset_cfg,
                    teacher_backbone=teacher_backbone,
                    tune_index=int(tune_index),
                    images_seen=int(images_seen),
                )
                all_rows.extend(rows)
                all_logs.update(logs)
            except Exception as exc:
                err_path = self.metrics_dir / f"tune_{int(tune_index):04d}_{dataset_name}_error.txt"
                err_path.write_text(traceback.format_exc(), encoding="utf-8")
                logger.error("[PathoROB] %s failed at tune_index %s: %s", dataset_name, int(tune_index), exc)

        return PluginResult(
            name=self.name,
            payload={"rows": all_rows},
            log_metrics=all_logs,
            selection_metric_value=None,
            selection_mode=None,
        )
