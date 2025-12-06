# Deployment Guide

Deploy GreenCologne to Hugging Face Spaces using Docker.

## Prerequisites
- Hugging Face account with token
- Docker installed (for local testing)
- Data uploaded to HF dataset

## Option 1: Docker Deployment (Recommended)

### Step 1: Create HF Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Settings:
   - **SDK**: Docker
   - **Visibility**: Public/Private
4. Clone the Space repository

### Step 2: Copy Files
```bash
git clone https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
cd YOUR_SPACE

# Copy from hf_space folder
cp -r /path/to/project/hf_space/* .
cp -r /path/to/project/hf_space/.streamlit .
```

### Step 3: Add Secrets
Go to Space **Settings** → **Repository secrets**:
- `HF_TOKEN`: Your Hugging Face token
- `DATASET_ID`: `Rahul-fix/cologne-green-data`

### Step 4: Push
```bash
git add .
git commit -m "Initial deployment"
git push
```

## Option 2: Streamlit SDK (Simpler)

1. Create Space with **Streamlit** SDK
2. Upload `app.py`, `utils.py`, `requirements.txt`
3. Add secrets in Settings

## Local Docker Testing

```bash
cd hf_space

# Build
docker build -t greencologne .

# Run (provide token)
docker run -p 7860:7860 -e HF_TOKEN="hf_xxx" greencologne

# Open http://localhost:7860
```

## Files Structure

```
hf_space/
├── Dockerfile          # Python 3.11 + GDAL
├── app.py              # Cloud app
├── utils.py            # Shared utilities
├── requirements.txt    # Dependencies
└── .streamlit/
    └── config.toml     # Theme config
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GDAL errors | Use Docker instead of Streamlit SDK |
| 401 Unauthorized | Regenerate HF token |
| HF_TOKEN missing | Add to Space Settings → Repository secrets |
