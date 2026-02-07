# I-JEPATH

Pathology-focused adaptation of I-JEPA for cross-resolution self-supervised learning on whole-slide images.

## Why this project
Standard SSL in pathology can overfit stain/scanner shortcuts. This project targets morphology-centric representations by predicting high-resolution target embeddings from lower-resolution tissue context (JEPA-style, no pixel reconstruction).

## Current scope
- Cross-resolution JEPA pretraining for pathology (`~1.0 mpp` context -> `~0.5 mpp` targets).
- Profile-aware anchor indexing from WSI + tissue masks.
- Online context/target extraction with target-footprint non-leakage masking.
- Smoke-trainable stage-1 pipeline and visualization QA.

## Example sample visualization
<img src="assets/preview/ijepath_flow.gif" alt="Cross-resolution context-to-target correspondence" width="920" />

Static companion previews:
- `assets/preview/ijepath_flow_all_steps.png`
- `assets/preview/ijepath_flow_zoom4.png`

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
python scripts/verify_pathology_training_runtime.py

# 2) Build slide metadata index
python scripts/build_slide_metadata_index_from_manifest.py \
  --manifest ${DATA_ROOT}/manifests/slides_with_tissue_masks.csv \
  --output ${DATA_ROOT}/indexes/slide_metadata_index.jsonl \
  --report ${DATA_ROOT}/indexes/slide_metadata_build_report.csv

# 3) Build profile-specific anchor catalog
python scripts/build_valid_context_anchor_catalog.py \
  --slide-index ${DATA_ROOT}/indexes/slide_metadata_index.jsonl \
  --profile configs/pathology_sampling_profile_ctx1p0_tgt0p5.yaml \
  --output ${DATA_ROOT}/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv

# 4) Pathology smoke training
python main.py --fname configs/pathology_tcga_prad_resampled2x_smoke.yaml --devices cuda:0
# (Edit paths inside the config to point to your ${DATA_ROOT})
```

## Preview generation
```bash
python scripts/preview_context_targets.py \
  --anchor-catalog ${DATA_ROOT}/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv \
  --output-dir outputs/pathology-previews \
  --num-samples 8
```

Outputs per sample:
- `preview_sXXX_<slide_id>_<anchor>.png`: static final flow layout (all steps visible).
- `ijepa_sXXX_<slide_id>_<anchor>.gif`: animated progressive reveal of the same layout.
- `ijepa_sXXX_<slide_id>_<anchor>_all_steps.png`: high-resolution final frame (`--final-png-scale` controls scale, default `2.0`).
- `zoom4_sXXX_<slide_id>_<anchor>.png`: zoomed 2x2 static view of steps 2-5 for a few anchors (`--num-zoomed-previews`, default `2`).

## Tests
```bash
pytest tests -k pathology
```

## Notes
- Metadata distinguishes `source_*_mpp` (pyramid read spacing) vs `output_*_mpp` (nominal requested spacing semantics).
- Tissue-aware target fallback policies are configurable (`skip_anchor`, `skip_slide`, `lower_threshold`).

## Upstream provenance
Based on the official I-JEPA implementation and paper:
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243.
