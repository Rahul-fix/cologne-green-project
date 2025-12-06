# GreenCologne Project Guides

This folder contains documentation for deploying and maintaining the GreenCologne dashboard.

## 📚 Available Guides

| Guide | Description |
|-------|-------------|
| [1. Deployment Guide](01_deployment.md) | Deploy to Hugging Face Spaces |
| [2. Data Update Guide](02_data_updates.md) | Update stats and sync to cloud |
| [3. Local Development](03_local_development.md) | Run and test locally |

## 🚀 Quick Start

### Run Locally
```bash
# Local version (uses local files)
uv run streamlit run streamlit_app/app_local.py

# Cloud version (uses Hugging Face data)
uv run streamlit run streamlit_app/app_hf.py
```

### Deploy to HF Spaces
```bash
cd hf_space
docker build -t greencologne .
docker run -p 7860:7860 -e HF_TOKEN="your_token" greencologne
```

### Sync New Data
```bash
uv run python scripts/06_calculate_full_stats.py
uv run python scripts/sync_to_hf.py
```
