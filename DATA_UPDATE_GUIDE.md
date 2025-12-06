# Data Update Guide

This guide explains how to update the application data when new satellite imagery or inference results (segmentation masks) become available.

## 1. Updating Statistics Only (Fast)
If you have generated new segmentation masks (`_mask.tif`) or have new raw tiles (`.jp2`) in your local folders and simply want to update the dashboard statistics:

```bash
uv run python scripts/06_calculate_full_stats.py
```

This script will:
1. Scan `data/raw` and `data/processed`.
2. Calculate Green Area and Land Cover distribution for every Veedel.
3. Aggregate the results into `data/stats/extended_stats.parquet`.
4. The Streamlit app will automatically read this new file on next reload.

## 2. Syncing Everything (Full Pipeline)
If you want to ensure your local environment is completely in sync with the Hugging Face repository and OpenNRW source (downloading missing files + calculating stats):

```bash
uv run python scripts/07_update_all_data.py
```

## 3. Generating Web-Optimized Tiles
If you have new imagery and want to ensure the map loads fast (generating `_optimized.tif`):

```bash
uv run python scripts/08_create_web_optimized_tiles.py
```

## verifying the Update
After running the stats script, restart your Streamlit app:
```bash
uv run streamlit run streamlit_app/app_local.py
```
Check the **Statistics** tab to see the updated figures.
