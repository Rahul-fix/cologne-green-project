from huggingface_hub import HfApi
from dotenv import load_dotenv
from pathlib import Path
import os

# Load Env
env_path = Path(__file__).parent.parent / "DL_cologne_green" / ".env"
load_dotenv(env_path)

token = os.getenv("HF_TOKEN")
dataset_id = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

if not token:
    print("Error: HF_TOKEN missing.")
    exit(1)

print(f"Syncing Local -> Remote ({dataset_id})...")
api = HfApi(token=token)

# 1. Upload Extended Stats
local_stats = Path("data/stats/extended_stats.parquet")
if local_stats.exists():
    print(f"Uploading {local_stats}...")
    api.upload_file(
        path_or_fileobj=local_stats,
        path_in_repo="data/stats/extended_stats.parquet",
        repo_id=dataset_id,
        repo_type="dataset"
    )
    print("Stats uploaded.")
else:
    print(f"Warning: {local_stats} not found locally.")

# 2. Upload Web Optimized Tiles
local_wo = Path("data/web_optimized")
if local_wo.exists():
    print(f"Uploading {local_wo} folder (this may take a while)...")
    api.upload_folder(
        folder_path=local_wo,
        path_in_repo="data/web_optimized",
        repo_id=dataset_id,
        repo_type="dataset",
        ignore_patterns=[".DS_Store"]
    )
    print("Web Optimized folder uploaded.")
else:
    print(f"Warning: {local_wo} folder not found locally.")

print("Sync Complete.")
