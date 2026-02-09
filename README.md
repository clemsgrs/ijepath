# I-JEPATH

Pathology-focused adaptation of I-JEPA for cross-resolution self-supervised learning on whole-slide images.

## Why this project
Standard SSL in pathology can overfit stain/scanner shortcuts. This project targets morphology-centric representations by predicting high-resolution target embeddings from lower-resolution tissue context (JEPA-style, no pixel reconstruction).

## Current scope
- Cross-resolution JEPA pretraining for pathology (`~1.0 mpp` context -> `~0.5 mpp` targets).
- Profile-aware anchor indexing from WSI + tissue masks.
- Online context/target extraction with target-footprint non-leakage masking.
- Smoke-trainable stage-1 pipeline and visualization QA.

## Docs
- `docs/pathology/README.md`
- `docs/pathology/config-reference.md`

## Sample curation pipeline
<img src="assets/preview/ijepath_flow.gif" alt="TCGA-HC-8257 anchor A001 cross-resolution flow" width="920" />

## Dataset structure
Use any dataset root; below is the expected structure and contracts:

```text
<DATA_ROOT>/
  manifests/
    slides_with_tissue_masks.csv
  indexes/
    slide_metadata_index.jsonl
    anchors_<profile>.csv
```

Manifest CSV format (`slides_with_tissue_masks.csv`):
- Header: `slide_id,wsi_path,mask_path`
- One row per slide.
- `wsi_path` and `mask_path` can point anywhere (absolute or relative paths). They do not need to be inside `<DATA_ROOT>`.

## Quick start
```bash
# Set your dataset root once
DATA_ROOT=/path/to/your-dataset

# 1) Verify runtime
python scripts/verify_training_runtime.py

# 2) Build slide metadata index
python scripts/build_slide_metadata_index_from_manifest.py \
  --manifest ${DATA_ROOT}/manifests/slides_with_tissue_masks.csv \
  --output ${DATA_ROOT}/indexes/slide_metadata_index.jsonl \
  --report ${DATA_ROOT}/indexes/slide_metadata_build_report.csv

# 3) Build profile-specific anchor catalog
python scripts/build_valid_context_anchor_catalog.py \
  --slide-index ${DATA_ROOT}/indexes/slide_metadata_index.jsonl \
  --profile configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml \
  --output ${DATA_ROOT}/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv

# 4) Smoke training (layered config: defaults + profile + run)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --profile-config configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml \
  --run-config configs/runs/tcga_prad_smoke.yaml \
  data.slide_manifest_csv=${DATA_ROOT}/manifests/slides_with_tissue_masks.csv \
  data.slide_metadata_index_jsonl=${DATA_ROOT}/indexes/slide_metadata_index.jsonl \
  data.anchor_catalog_csv=${DATA_ROOT}/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv

# defaults config is implicit: configs/defaults.yaml

# Merged resolved config is saved automatically to:
# outputs/<run-folder>/params-ijepa.yaml
# Training step CSV columns are standardized:
# epoch, iteration, loss, context_keep_tokens, target_predict_tokens,
# iteration_time_ms, learning_rate, weight_decay

# Optional W&B logging (single log payload per epoch, x-axis=epoch):
# ... wandb.enable=true wandb.project=<project> wandb.username=<entity>
# Optional terminal/checkpoint logging controls:
# ... logging.step_log_every_iters=0 logging.checkpoint_every_epochs=50

# Epoch semantics:
# - data.samples_per_epoch=null  -> one full pass on anchor_catalog rows
# - data.samples_per_epoch=<int> -> fixed virtual epoch length (e.g. smoke runs)
```

## Preview generation
```bash
python scripts/preview_context_targets.py \
  --anchor-catalog ${DATA_ROOT}/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv \
  --output-dir outputs/previews \
  --num-samples 8
```

Outputs per sample:
- `preview_sXXX_<slide_id>_<anchor>.png`: static final flow layout (all steps visible).
- `ijepa_sXXX_<slide_id>_<anchor>.gif`: animated progressive reveal of the same layout.
- `ijepa_sXXX_<slide_id>_<anchor>_all_steps.png`: high-resolution final frame (`--final-png-scale` controls scale, default `2.0`).
- `zoom4_sXXX_<slide_id>_<anchor>.png`: zoomed 2x2 static view of steps 2-5 for a few anchors (`--num-zoomed-previews`, default `2`).

## Tests
```bash
pytest tests
pytest -m integration tests/test_pipeline_integration.py
```

## Notes
- Metadata distinguishes `source_*_mpp` (pyramid read spacing) vs `output_*_mpp` (nominal requested spacing semantics).
- Tissue-aware target fallback policies are configurable (`skip_anchor`, `skip_slide`, `lower_threshold`).
- Target box patch-alignment is configurable via `data.align_targets_to_patch_grid`.

### Target Sampling Strategy: Aligned vs Non-Aligned
Use `data.align_targets_to_patch_grid` to choose how target boxes are sampled in context coordinates.

- Notation used below:
  - `C`: continuous target content extent in context coordinates.
  - `R`: raw tokenized footprint before collator truncation.
  - `T`: post-collator predictor footprint (`T ⊆ R`).

- Non-aligned (`false`, default):
  - Preserves sub-patch offsets and increases spatial diversity.
  - Floor/ceil rasterization can make `R` larger than `C`.
  - After `min_keep_pred` truncation, `T` can drop part of `R`.
  - Net effect can include both `C \ T` (target content not represented in predictor footprint) and `T \ C` (predictor footprint outside target content).
  - Leakage clarification: context masking is built from pre-truncation `R` (typically `R ⊇ C`), not from `T`, so this is conservative over-masking (context sees less), not target exposure.
  - Practical consequence: the core issue is supervision mismatch, not leakage; predictor summarizes tokens indexed by `T`, while teacher supervision comes from the full target crop embedding.

![Non-aligned target path](assets/pathology/stagec_non_aligned_path.png)

- Patch-aligned (`true`):
  - Snaps boxes to patch boundaries, producing cleaner and more interpretable footprint geometry.
  - For the standard fixed-size setup here (target size is patch-multiple), alignment makes `C`, `R`, and `T` coincide and removes `C/T` mismatch.
  - Trades off some spatial diversity (fewer sub-patch offsets).

![Patch-aligned target path](assets/pathology/stagec_aligned_path.png)

- Practical guidance:
  - Prefer `false` for representation diversity.
  - Prefer `true` for debugging, controlled ablations, and maximum predictor-teacher geometric alignment.

## Upstream provenance
Based on the official I-JEPA implementation and paper:
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243.
