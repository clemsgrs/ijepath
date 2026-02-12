from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch

from .plugins import BenchmarkPlugin, PathoROBPlugin

logger = logging.getLogger("ijepath")

_EARLY_STOPPING_METRIC_MODES: dict[str, dict[str, str]] = {
    "pathorob": {
        "ri": "max",
        "clustering_score": "max",
        "apd_id": "min",
        "apd_ood": "min",
        "apd_avg": "min",
    }
}


class Tuner:
    """Orchestrator for robustness/downstream tuning plugins."""

    def __init__(self, cfg: dict, device: torch.device, output_dir: Path):
        self.cfg = dict(cfg or {})
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.plugins: list[BenchmarkPlugin] = []
        self.primary_plugin_name: str | None = None
        self._register_plugins()
        self.selection_cfg = self._resolve_selection_cfg()

        for plugin in self.plugins:
            plugin.bind_runtime(self.output_dir)

    def _register_plugins(self) -> None:
        plugin_cfgs = list(self.cfg.get("plugins", []))

        for raw in plugin_cfgs:
            p = dict(raw or {})
            if not bool(p.get("enable", True)):
                continue
            ptype = str(p.get("type", "")).strip().lower()
            if ptype == "pathorob":
                plugin = PathoROBPlugin(p, self.device, output_dir=self.output_dir)
                self.plugins.append(plugin)
            else:
                raise ValueError(f"Unknown tuning plugin type: {ptype}")

    def _resolve_selection_cfg(self) -> dict | None:
        early_cfg = dict(self.cfg.get("early_stopping", {}) or {})
        if not bool(early_cfg.get("enable", False)):
            return None

        selection = dict(early_cfg.get("selection", {}) or {})
        plugin = str(selection.get("plugin", "")).strip().lower()
        dataset = str(selection.get("dataset", "")).strip()
        metric = str(selection.get("metric", "")).strip()
        mode = _EARLY_STOPPING_METRIC_MODES[plugin][metric]
        self.primary_plugin_name = plugin
        return {
            "plugin": plugin,
            "dataset": dataset,
            "metric": metric,
            "mode": mode,
            "log_key": f"{plugin}/{dataset}/{metric}",
        }

    @torch.no_grad()
    def tune(self, teacher, tune_index: int, images_seen: int) -> dict:
        out = {
            "plugins": {},
            "log_metrics": {},
            "selection": None,
        }

        for plugin in self.plugins:
            if not plugin.should_run(images_seen=int(images_seen), tune_index=int(tune_index)):
                continue

            result = plugin.run(teacher=teacher, tune_index=int(tune_index), images_seen=int(images_seen))
            out["plugins"][plugin.name] = result.payload

            for key, value in result.log_metrics.items():
                out["log_metrics"][f"{plugin.name}/{key}"] = float(value)

        if self.selection_cfg is not None:
            value = out["log_metrics"].get(self.selection_cfg["log_key"])
            if value is None:
                raise ValueError(
                    "Configured early-stopping target was not emitted in this tuning event: "
                    f"{self.selection_cfg['log_key']}. Available metrics: {sorted(out['log_metrics'].keys())}"
                )
            out["selection"] = {
                "plugin": self.selection_cfg["plugin"],
                "dataset": self.selection_cfg["dataset"],
                "metric": self.selection_cfg["metric"],
                "metric_value": float(value),
                "mode": self.selection_cfg["mode"],
            }

        self._persist_unified_metrics(
            results=out,
            tune_index=int(tune_index),
            images_seen=int(images_seen),
        )
        return out

    def get_log_metrics(self, results: dict) -> dict[str, float]:
        return dict(results.get("log_metrics", {}))

    def get_selection(self, results: dict) -> dict | None:
        return results.get("selection")

    def get_selection_mode(self) -> str | None:
        if self.selection_cfg is None:
            return None
        return str(self.selection_cfg["mode"])

    def _persist_unified_metrics(self, results: dict, tune_index: int, images_seen: int) -> None:
        rows = []
        plugin_payloads = dict(results.get("plugins", {}))
        for plugin_name, payload in plugin_payloads.items():
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                for row in payload["rows"]:
                    row = dict(row)
                    row.setdefault("plugin", plugin_name)
                    row.setdefault("tune_index", int(tune_index))
                    row.setdefault("images_seen", int(images_seen))
                    rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        out_csv = self.metrics_dir / f"tune_{int(tune_index):04d}.csv"
        df.to_csv(out_csv, index=False)

        roll_jsonl = self.metrics_dir / "all_metrics.jsonl"
        with roll_jsonl.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True))
                handle.write("\n")

        logger.info(
            "Persisted unified tuning metrics: rows=%d tune_index=%d images_seen=%d",
            len(df),
            int(tune_index),
            int(images_seen),
        )
