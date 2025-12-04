# VM Inference Setup Guide

This guide describes how to set up a Virtual Machine (VM) to run the full Cologne Green inference pipeline using GPU acceleration.

## 1. Prerequisites
- **VM Instance:** A machine with a GPU (e.g., NVIDIA T4, A10, or V100).
- **OS:** Ubuntu 20.04 or 22.04 LTS recommended.
- **Storage:** At least 100GB of disk space (for raw images and output).

## 2. Initial Setup

### Clone the Repository
```bash
git clone https://github.com/Rahul-fix/cologne-green-project.git
cd cologne-green-project
git checkout dl-pipeline

# Initialize submodule (FLAIR-HUB)
git submodule update --init --recursive
```

### Install Python & Dependencies (GDAL)
`rasterio` and `geopandas` require system-level GDAL libraries.
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev build-essential python3-gdal
```

### Install Python & Dependencies
We use `uv` for fast dependency management, but standard `pip` works too.

**Option A: Using `uv` (Recommended)**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Sync environment
uv sync
```

**Option B: Using `pip`**
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Install local package in editable mode
pip install -e .
```

### Install FLAIR-HUB
You need to install the `flairhub` package from the local submodule.
```bash
uv pip install -e DL_cologne_green/FLAIR-HUB
# OR
pip install -e DL_cologne_green/FLAIR-HUB
```

**Option C: Setup with Python 3.11 (via uv)**
Recommended if your system Python is older than 3.11 (e.g., Ubuntu 20.04).

1. **Install Python 3.11 and Create Virtual Environment**
   ```bash
   uv python install 3.11
   uv venv .venv311 --python 3.11
   ```

2. **Install Dependencies**
   ```bash
   # Activate the environment
   source .venv311/bin/activate
   
   # Install the project in editable mode
   # We use uv pip to ensure it installs into the active venv
   uv pip install -p .venv311 -e DL_cologne_green/FLAIR-HUB
   ```

## 3. Download Data & Weights

### Download Model Weights
This downloads the pre-trained weights (~800MB) to `DL_cologne_green/`.
```bash
uv run python scripts/download_weights.py
# OR if using pip
python scripts/download_weights.py
```

### Download Boundaries (Important!)
Before downloading tiles, you need the Cologne city boundaries.
```bash
uv run python scripts/download_boundaries.py
```

### Download Satellite Tiles
This script first identifies all tiles covering Cologne (using the boundaries) and then downloads them.
It automatically prefers the latest available year (e.g., 2025) but falls back to older years (2023) if necessary.

1. **Generate Tile List**
   ```bash
   uv run python scripts/find_cologne_tiles.py
   ```

2. **Download Tiles**
   This downloads ~20GB of data to `data/raw/`.
   ```bash
   # Default uses 8 parallel workers
   uv run python scripts/download_all_tiles.py --workers 16
   # OR
   python scripts/download_all_tiles.py --workers 16
   ```

## 4. Run Inference (GPU Enabled)

We have prepared a specific configuration for VM usage: `DL_cologne_green/config_vm_inference.yaml`.
- **GPU:** Enabled (`use_gpu: True`)
- **Batch Size:** 16 (Optimized for Nvidia T4 with 16GB VRAM)
- **Workers:** 4 (Matches your 4 vCPUs)

Run the processing script pointing to this config:

```bash
uv run python scripts/02_process_local.py --config DL_cologne_green/config_vm_inference.yaml
# OR
python scripts/02_process_local.py --config DL_cologne_green/config_vm_inference.yaml
```

### Performance Tuning
For your specific setup (4 vCPU, 15GB RAM, T4 GPU):
- **Batch Size:** `10` is a safe starting point (verified working). `16` might work but could cause OOM.
- **Download Workers:** Use `--workers 16` for downloads to saturate your network connection.
- **Inference Workers:** Set `num_worker: 0` (main process only) to avoid System RAM OOM. Increasing this consumes significant RAM per worker.

### Monitoring
- The script will process images one by one.
- Outputs (masks and NDVI) will be saved to `data/processed/`.
- Logs are printed to the console.

## 5. Transfer Results

### Option A: Download Tarball
Once finished, you can compress and download the `data/processed` folder.
```bash
tar -czvf cologne_green_processed.tar.gz data/processed
```

### Option B: Sync to Google Cloud Storage (GCS)
If you have a GCS bucket (e.g., `gs://cologne-green-project`), you can sync results directly.

1. **Authenticate (if needed)**
   If your VM doesn't have Storage scopes, login:
   ```bash
   gcloud auth login
   ```

2. **Run Sync Script**
   This script organizes files into `processed/masks`, `processed/ndvi`, etc.
   ```bash
   uv run python scripts/sync_to_gcs.py cologne-green-project
   # OR
   python scripts/sync_to_gcs.py cologne-green-project
   ```

3. **Verify**
   ```bash
   gsutil ls -r gs://cologne-green-project/
   ```

### Option C: Upload to Hugging Face (Spaces/Datasets)
You can upload the processed data directly to a Hugging Face Dataset, which can then be used by your Spaces app.

1. **Install Dependencies**
   ```bash
   uv add huggingface_hub
   # OR
   pip install huggingface_hub
   ```

2. **Get your HF Token**
   - Go to [HF Settings > Tokens](https://huggingface.co/settings/tokens).
   - Create a **Write** token.

3. **Run Upload Script**
   You can pass the token directly or set it as `HF_TOKEN` in `.env`.
   ```bash
   # Replace YOUR_WRITE_TOKEN with your actual token
   uv run python scripts/upload_to_hf.py --token YOUR_WRITE_TOKEN --auto
   
   # OR interactively
   uv run python scripts/upload_to_hf.py
   ```
   This will upload `data/raw`, `data/processed`, `data/stats`, and `data/boundaries` to a new dataset (default: `your-username/cologne-green-data`).
