# 🌿 GreenCologne: Cloud-Native Implementation Guide (v3.0)
**Status:** Production-Ready | **Architecture:** Serverless + DuckDB | **Last Updated:** December 2025

---

## 📋 Overview
This guide transitions your local GreenCologne project to a robust, scalable Google Cloud Platform (GCP) architecture.

**New in v3.0:**
- **Hugging Face Spaces Support:** Deploy for free on HF Spaces.
- **Improved Data Structure:** Organized data folders (`raw`, `boundaries`, `metadata`).

---

## 🚀 Phase 1: Cloud Infrastructure Setup
**Goal:** Create the project and storage buckets.

1.  **Create GCP Project:**
    *   Name: `cologne-green-project`
    *   Enable APIs: Compute Engine, Cloud Run, Artifact Registry, Cloud Build.

2.  **Create Storage Bucket:**
    ```bash
    export BUCKET_NAME="cologne-green-data-v1"
    gcloud storage buckets create gs://$BUCKET_NAME --location=europe-west3
    ```

---

## 📦 Phase 2: Full-Scale Data Pipeline
**Goal:** Download ALL satellite tiles and prepare the model.

### 1. Download All Satellite Tiles
```bash
# Identify tiles (uses data/metadata/dop_nw.csv)
uv run scripts/find_cologne_tiles.py

# Download tiles (to data/raw/)
uv run scripts/download_all_tiles.py

# Upload to GCS
gsutil -m cp data/raw/*.jp2 gs://$BUCKET_NAME/raw/
```

### 2. Upload Boundaries & Metadata
```bash
# Upload Boundaries
gsutil cp data/boundaries/*.parquet gs://$BUCKET_NAME/data/boundaries/

# Upload Metadata (Optional)
gsutil cp data/metadata/*.csv gs://$BUCKET_NAME/data/metadata/
```

### 3. FLAIR-HUB Setup (The Right Way)
1.  **Clone as Submodule:**
    ```bash
    mkdir -p DL_cologne_green
    cd DL_cologne_green
    git submodule add https://github.com/flair-hub/flair-hub.git FLAIR-HUB
    pip install -e FLAIR-HUB
    ```

2.  **Download Model Weights:**
    ```bash
    uv run scripts/download_weights.py
    ```

---

## ⚙️ Phase 3: Cloud Processing Pipeline
**Goal:** Run FLAIR-HUB inference on a Deep Learning VM.

1.  **Create Deep Learning VM (T4 GPU):**
    ```bash
    gcloud compute instances create green-cologne-gpu \
        --zone=europe-west3-b \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True"
    ```

2.  **Deploy & Run:**
    ```bash
    # Zip code
    zip -r pipeline_code.zip scripts/ DL_cologne_green/ pyproject.toml

    # Copy & SSH
    gcloud compute scp pipeline_code.zip green-cologne-gpu:~
    gcloud compute ssh green-cologne-gpu

    # On VM:
    unzip pipeline_code.zip
    pip install uv && uv sync
    uv run scripts/download_weights.py # Get weights on VM
    uv run scripts/02_process_local.py # Run batch inference (reads data/raw, writes data/processed)
    gsutil -m cp data/processed/*.tif gs://$BUCKET_NAME/processed/
    ```

---

## 🌐 Phase 4: Deployment Options

### Option A: Google Cloud Run (Scalable, Production)
*Best for: High traffic, auto-scaling, private deployment.*
See previous guide sections for Dockerfile and `gcloud run deploy`.

### Option B: Hugging Face Spaces (Free, Easy)
*Best for: Demos, public sharing, zero cost.*

1.  **Create Space:**
    - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    - Create new Space -> SDK: **Streamlit**.

2.  **Setup Secrets:**
    - Go to **Settings** -> **Variables and secrets**.
    - Add: `HMAC_KEY`, `HMAC_SECRET`, `BUCKET_NAME`.

3.  **Deploy Code:**
    Upload:
    - `streamlit_app/app_hf.py` (Rename to `app.py`)
    - `requirements.txt`
    - `packages.txt`

    **`requirements.txt`:**
    ```text
    streamlit
    duckdb
    pandas
    geopandas
    shapely
    folium
    streamlit-folium
    plotly
    matplotlib
    rasterio
    google-cloud-storage
    ```

    **`packages.txt`:**
    ```text
    gdal-bin
    libgdal-dev
    ```
