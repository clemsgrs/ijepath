from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from .plugins import BenchmarkPlugin, PathoROBPlugin

logger = logging.getLogger("ijepath")


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

        for plugin in self.plugins:
            plugin.bind_runtime(self.output_dir)

    def _register_plugins(self) -> None:
        plugin_cfgs = list(self.cfg.get("plugins", []))
        selected_primary = []

        for raw in plugin_cfgs:
            p = dict(raw or {})
            if not bool(p.get("enable", True)):
                continue
            ptype = str(p.get("type", "")).strip().lower()
            if ptype == "pathorob":
                plugin = PathoROBPlugin(p, self.device, output_dir=self.output_dir)
                self.plugins.append(plugin)
                if bool(p.get("use_for_early_stopping", False)):
                    selected_primary.append(plugin.name)
            else:
                raise ValueError(f"Unknown tuning plugin type: {ptype}")

        if len(selected_primary) > 1:
            raise ValueError(
                "At most one enabled plugin may set use_for_early_stopping=true"
            )
        if selected_primary:
            self.primary_plugin_name = selected_primary[0]

    @torch.no_grad()
    def tune(self, teacher, eval_index: int, images_seen: int) -> dict:
        out = {
            "plugins": {},
            "log_metrics": {},
            "selection": None,
        }

        for plugin in self.plugins:
            if not plugin.should_run(images_seen=int(images_seen), eval_index=int(eval_index)):
                continue

            result = plugin.run(teacher=teacher, eval_index=int(eval_index), images_seen=int(images_seen))
            out["plugins"][plugin.name] = result.payload

            for key, value in result.log_metrics.items():
                out["log_metrics"][f"{plugin.name}/{key}"] = float(value)

            if self.primary_plugin_name == plugin.name:
                if result.selection_metric_value is None:
                    raise ValueError(
                        f"Primary plugin '{plugin.name}' did not provide selection_metric_value "
                        "for this evaluation event."
                    )
                out["selection"] = {
                    "plugin": plugin.name,
                    "metric_value": result.selection_metric_value,
                    "mode": result.selection_mode,
                }

        self._persist_unified_metrics(
            results=out,
            eval_index=int(eval_index),
            images_seen=int(images_seen),
        )
        return out

    def get_log_metrics(self, results: dict) -> dict[str, float]:
        return dict(results.get("log_metrics", {}))

    def get_selection(self, results: dict) -> dict | None:
        return results.get("selection")

    def _persist_unified_metrics(self, results: dict, eval_index: int, images_seen: int) -> None:
        rows = []
        plugin_payloads = dict(results.get("plugins", {}))
        for plugin_name, payload in plugin_payloads.items():
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                for row in payload["rows"]:
                    row = dict(row)
                    row.setdefault("plugin", plugin_name)
                    row.setdefault("eval_index", int(eval_index))
                    row.setdefault("images_seen", int(images_seen))
                    rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        out_csv = self.metrics_dir / f"eval_{int(eval_index):04d}.csv"
        df.to_csv(out_csv, index=False)

        roll_csv = self.metrics_dir / "all_metrics.csv"
        if roll_csv.exists():
            old = pd.read_csv(roll_csv)
            pd.concat([old, df], axis=0).reset_index(drop=True).to_csv(roll_csv, index=False)
        else:
            df.to_csv(roll_csv, index=False)

        logger.info(
            "Persisted unified tuning metrics: rows=%d eval_index=%d images_seen=%d",
            len(df),
            int(eval_index),
            int(images_seen),
        )
