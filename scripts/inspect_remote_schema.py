import duckdb
from huggingface_hub import HfFileSystem
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd

# Load Env
env_path = Path(__file__).parent.parent / "DL_cologne_green" / ".env"
load_dotenv(env_path)

token = os.getenv("HF_TOKEN")
dataset_id = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")
base_url = f"hf://datasets/{dataset_id}"

print(f"Inspecting Remote Assets for: {dataset_id}")

# 1. Check CSV Schema
print(f"\n--- Schema: cologne_tiles.csv ---")
try:
    df_csv = pd.read_csv(f"{base_url}/data/metadata/cologne_tiles.csv", storage_options={"token": token})
    print(df_csv.columns.tolist())
except Exception as e:
    print(f"Error reading CSV: {e}")

# 2. Check Web Optimized Folder
print(f"\n--- Checking Web Optimized Folder ---")
if token:
    fs = HfFileSystem(token=token)
    try:
        wo = fs.glob(f"datasets/{dataset_id}/data/web_optimized/*")
        print(f"Web Optimized Files Found: {len(wo)}")
    except Exception as e:
        print(f"Error listing web_optimized: {e}")

# 3. Check Parquet Schemas
con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("INSTALL httpfs; LOAD httpfs;")
if token:
    con.register_filesystem(fs)

files_to_check = [
    f"{base_url}/data/boundaries/Stadtviertel.parquet",
    f"{base_url}/data/stats/extended_stats.parquet"
]

for f in files_to_check:
    print(f"\n--- Schema: {f} ---")
    try:
        df = con.execute(f"DESCRIBE SELECT * FROM '{f}'").df()
        print(df[['column_name', 'column_type']])
    except Exception as e:
        print(f"Error: {e}")
