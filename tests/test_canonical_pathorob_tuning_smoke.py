from pathlib import Path

import torch

from ijepath import train_canonical_jepa as canonical_train


class _FakeDataset:
    total_anchors = 8

    def __len__(self):
        return 8

    def set_pass_index(self, _pass_index: int):
        return None


class _FakeLoader:
    def __len__(self):
        return 1

    def __iter__(self):
        batch_data = {
            "image": torch.rand(1, 3, 16, 16),
            "sample_metadata": [{}],
        }
        masks_enc = [torch.tensor([[0]], dtype=torch.long)]
        masks_pred = [torch.tensor([[0]], dtype=torch.long)]
        yield batch_data, masks_enc, masks_pred


class _FakeTuner:
    def __init__(self, cfg, device, output_dir):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

    def get_selection_mode(self):
        return "max"


class _FakeAsyncRuntime:
    submitted = []

    def __init__(self, tuning_cfg, teacher_template, output_dir):
        self.tuning_cfg = tuning_cfg
        self.teacher_template = teacher_template
        self.output_dir = output_dir
        self._pending = []

    def submit(self, tune_index, images_seen, snapshot_path):
        payload = {
            "tune_index": int(tune_index),
            "images_seen": int(images_seen),
            "snapshot_path": str(snapshot_path),
        }
        _FakeAsyncRuntime.submitted.append(payload)
        self._pending.append(payload)
        return {"queue_depth": len(self._pending), "dropped_tunes": 0}

    def poll_completed(self):
        completed = []
        while self._pending:
            event = self._pending.pop(0)
            completed.append(
                {
                    "tune_index": int(event["tune_index"]),
                    "images_seen": int(event["images_seen"]),
                    "result": {
                        "log_metrics": {"pathorob/camelyon/ri": 0.5},
                    },
                    "latency_seconds": 0.01,
                    "worker_seconds": 0.01,
                }
            )
        return completed

    def queue_depth(self):
        return len(self._pending)

    def dropped_tunes(self):
        return 0

    def shutdown(self, wait=False, timeout_s=2.0):
        return None


def test_canonical_training_with_async_tuning_smoke(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        canonical_train,
        "make_canonical_loader",
        lambda **kwargs: (_FakeDataset(), _FakeLoader(), None),
    )
    monkeypatch.setattr("ijepath.eval.tuner.Tuner", _FakeTuner)
    monkeypatch.setattr("ijepath.eval.async_runtime.AsyncTuningRuntime", _FakeAsyncRuntime)

    args = {
        "pretraining": {"mode": "canonical"},
        "data": {
            "batch_size_per_gpu": 1,
            "pin_mem": False,
            "num_workers": 0,
            "seed": 0,
            "wsi_backend": "asap",
            "slide_manifest_csv": "dummy.csv",
            "slide_metadata_parquet": "dummy.parquet",
            "anchor_catalog_manifest": "dummy_manifest.json",
            "sampling_strategy": "stratified_weighted",
            "sampling_stratum_key": "organ",
            "sampling_stratum_weights": "inverse_frequency",
            "persistent_workers": False,
            "prefetch_factor": 2,
            "max_open_slides_per_worker": 2,
            "anchor_stream_batch_size": 16,
            "spacing_tolerance": 0.05,
            "low_anchor_pass_warning_threshold": 1.0,
            "high_anchor_pass_warning_threshold": 5.0,
        },
        "canonical": {
            "input_mpp": 0.5,
            "source_tile_size_px": 16,
            "crop_size_px": 16,
            "transform_preset": "official_ijepa",
            "enc_mask_scale": [0.85, 1.0],
            "pred_mask_scale": [0.15, 0.2],
            "aspect_ratio": [0.75, 1.5],
            "num_enc_masks": 1,
            "num_pred_masks": 1,
            "min_keep": 1,
        },
            "meta": {
                "load_checkpoint": False,
                "architecture": "vit_tiny",
                "patch_size": 16,
                "pred_depth": 1,
                "pred_emb_dim": 192,
                "read_checkpoint": None,
                "use_bfloat16": False,
            },
        "optimization": {
            "ema": [0.996, 1.0],
            "total_images_budget": 1,
            "final_lr": 1.0e-06,
            "final_weight_decay": 0.1,
            "ipe_scale": 1.0,
            "lr": 0.0002,
            "start_lr": 0.00002,
            "warmup": 0.0,
            "weight_decay": 0.04,
        },
        "training": {
            "log_every": 1,
            "save_every": 1,
        },
        "tuning": {
            "enable": True,
            "seed": 0,
            "tune_every": 1,
            "run_baseline_at_zero": True,
            "execution": {
                "mode": "async",
                "device": "auto",
                "max_pending_jobs": 2,
                "coalesce_policy": "newest",
                "poll_every_steps": 1,
                "fail_on_backlog": False,
                "keep_last_n_snapshots": 2,
            },
            "plugins": [],
            "early_stopping": {"enable": False},
        },
        "output": {
            "root": str(tmp_path / "outputs"),
            "shared_cache_root": str(tmp_path / "outputs" / "cache"),
        },
        "logging": {
            "write_tag": "canonical-tuning-smoke",
            "step_log_every_images": 1,
            "performance_debug": {"enable": False},
        },
        "wandb": {
            "enable": False,
        },
    }

    _FakeAsyncRuntime.submitted.clear()
    canonical_train.main(args=args, resume_preempt=False, distributed_state=(1, 0))

    assert len(_FakeAsyncRuntime.submitted) >= 1
