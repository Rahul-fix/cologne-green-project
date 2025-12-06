# Data Update Guide

How to update statistics and sync data to Hugging Face when you have new imagery or inference results.

## Workflow Overview

```
Local Inference → Update Stats → Create Optimized Tiles → Upload to HF
```

## Step 1: Generate New Masks

Run your FLAIR inference pipeline to create segmentation masks:

```bash
# Your inference script (example)
uv run python scripts/run_flair_inference.py
```

This creates `*_mask.tif` files in `data/processed/`.

## Step 2: Update Statistics

Recalculate stats for all Veedels:

```bash
uv run python scripts/06_calculate_full_stats.py
```

This updates:
- `data/stats/extended_stats.parquet` (NDVI, green area, land cover per Veedel)

## Step 3: Create Web-Optimized Tiles

For faster map loading:

```bash
uv run python scripts/08_create_web_optimized_tiles.py
```

This creates optimized tiles in `data/web_optimized/`.

## Step 4: Upload to Hugging Face

Sync new data to the cloud:

```bash
# Using sync script
uv run python scripts/sync_to_hf.py

# Or using CLI
huggingface-cli upload Rahul-fix/cologne-green-data data/stats/ data/stats/ --repo-type dataset
huggingface-cli upload Rahul-fix/cologne-green-data data/web_optimized/ data/web_optimized/ --repo-type dataset
```

## Step 5: Verify

```bash
# Check remote files
uv run python scripts/inspect_hf_repo.py

# Test cloud app locally
uv run streamlit run streamlit_app/app_hf.py
```

## Quick Reference

| Task | Command |
|------|---------|
| Update stats | `uv run python scripts/06_calculate_full_stats.py` |
| Create optimized tiles | `uv run python scripts/08_create_web_optimized_tiles.py` |
| Sync to HF | `uv run python scripts/sync_to_hf.py` |
| Check HF files | `uv run python scripts/inspect_hf_repo.py` |

## Files That Need Syncing

| Local File | Remote Path |
|------------|-------------|
| `data/stats/extended_stats.parquet` | `data/stats/extended_stats.parquet` |
| `data/web_optimized/*.tif` | `data/web_optimized/*.tif` |
| `data/processed/*_mask.tif` | `data/processed/*_mask.tif` |
