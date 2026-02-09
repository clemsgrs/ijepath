"""Render notebook figures to disk for fast Stage C iteration.

This is the versioned counterpart of the temporary preview renderer so the
figure-generation workflow is reproducible from the repository.
"""

import json
import os
from pathlib import Path

# Keep matplotlib caches writable in tmp for reproducible local rendering.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_PATH = Path(
    os.environ.get(
        "STAGEC_NOTEBOOK",
        str(REPO_ROOT / "notebooks/test-fixture-cross-resolution-pipeline.ipynb"),
    )
)
OUT_ROOT = Path(os.environ.get("STAGEC_OUT_ROOT", "output/stagec-preview"))
OUT_DIR_OVERRIDE = os.environ.get("STAGEC_OUT_DIR")
LAST_CELL = int(os.environ.get("STAGEC_LAST_CELL", "30"))

if OUT_DIR_OVERRIDE:
    OUT_DIR = Path(OUT_DIR_OVERRIDE)
else:
    OUT_DIR = OUT_ROOT

OUT_DIR.mkdir(parents=True, exist_ok=True)
nb = json.loads(NB_PATH.read_text())
ns = {"__name__": "__main__"}
counter = 0


def save_all_open_figures():
    global counter
    for n in list(plt.get_fignums()):
        fig = plt.figure(n)
        counter += 1
        out = OUT_DIR / f"{counter:02d}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close("all")


plt.show = save_all_open_figures

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    stripped = src.lstrip()
    if stripped.startswith("!") or stripped.startswith("%"):
        continue
    exec(compile(src, f"cell_{i}", "exec"), ns, ns)
    if i >= LAST_CELL:
        break

print(f"notebook={NB_PATH}")
print(f"out_dir={OUT_DIR}")
print(f"rendered {counter} figure(s)")
