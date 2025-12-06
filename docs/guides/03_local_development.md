# Local Development Guide

How to run and develop the GreenCologne dashboard locally.

## Prerequisites

- Python 3.11+
- uv package manager
- HF token (for cloud features)

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USER/cologne-green-project
cd cologne-green-project
uv sync

# Set environment
cp DL_cologne_green/.env.example DL_cologne_green/.env
# Edit .env and add HF_TOKEN
```

## Running Locally

### Local Version (Uses Local Files)
```bash
uv run streamlit run streamlit_app/app_local.py
# Opens http://localhost:8501
```

### Cloud Version (Uses HF Data)
```bash
uv run streamlit run streamlit_app/app_hf.py
# Opens http://localhost:8501
```

### Docker Version
```bash
cd hf_space
docker build -t greencologne .
docker run -p 7860:7860 -e HF_TOKEN="hf_xxx" greencologne
# Opens http://localhost:7860
```

## Code Structure

```
streamlit_app/
├── app_local.py    # Local file version
├── app_hf.py       # Cloud/HF version
└── utils.py        # Shared utilities

hf_space/           # Ready-to-deploy HF Space
├── Dockerfile
├── app.py
├── utils.py
└── requirements.txt
```

## Keeping Versions in Sync

When editing `app_local.py`, apply changes to `app_hf.py`:

```bash
# 1. Edit app_local.py
# 2. Copy changes to app_hf.py (adapt data loading)
# 3. Update hf_space/app.py for deployment
cp streamlit_app/app_hf.py hf_space/app.py
cp streamlit_app/utils.py hf_space/utils.py
```

### Key Differences

| Feature | app_local.py | app_hf.py |
|---------|--------------|-----------|
| Data source | Local parquet files | HF Dataset via DuckDB |
| Token | Not required | Required (HF_TOKEN) |
| Tile access | Direct file read | HfFileSystem |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/06_calculate_full_stats.py` | Recalculate Veedel statistics |
| `scripts/08_create_web_optimized_tiles.py` | Create optimized map tiles |
| `scripts/sync_to_hf.py` | Upload data to Hugging Face |
| `scripts/inspect_hf_repo.py` | Check remote file structure |
