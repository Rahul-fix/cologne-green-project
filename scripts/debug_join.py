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

print(f"Debugging Data Join for: {dataset_id}")

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("INSTALL httpfs; LOAD httpfs;")
if token:
    fs = HfFileSystem(token=token)
    con.register_filesystem(fs)

# 1. Check Names in Boundaries
print("\n--- Boundaries Names ---")
try:
    df_b = con.execute(f"SELECT name FROM '{base_url}/data/boundaries/Stadtviertel.parquet' LIMIT 5").df()
    print(df_b)
except Exception as e:
    print(f"Error reading boundaries: {e}")

# 2. Check Names in Stats
print("\n--- Stats Names ---")
try:
    df_s = con.execute(f"SELECT name, ndvi_mean FROM '{base_url}/data/stats/extended_stats.parquet' LIMIT 5").df()
    print(df_s)
except Exception as e:
    print(f"Error reading stats: {e}")

# 3. Test Join
print("\n--- Testing Join ---")
try:
    query = f"""
        SELECT v.name, s.ndvi_mean, s.green_area_m2
        FROM '{base_url}/data/boundaries/Stadtviertel.parquet' v 
        JOIN '{base_url}/data/stats/extended_stats.parquet' s ON v.name = s.name
    """
    df_join = con.execute(query).df()
    print(f"Joined Rows: {len(df_join)}")
    if not df_join.empty:
        print("Sample Joined Data:")
        print(df_join.head(3))
    else:
        print("JOIN RESULTED IN EMPTY DATAFRAME!")
except Exception as e:
    print(f"Join Error: {e}")
