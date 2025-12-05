import duckdb
from huggingface_hub import HfFileSystem
from dotenv import load_dotenv
import os

load_dotenv("DL_cologne_green/.env")
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

BASE_URL = f"hf://datasets/{DATASET_ID}"
STATS_FILE = f"{BASE_URL}/data/stats/stats.parquet"
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"

def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    fs = HfFileSystem(token=HF_TOKEN)
    con.register_filesystem(fs)
    
    print(f"Querying {DISTRICTS_FILE}...")
    try:
        count = con.execute(f"SELECT count(*) FROM '{DISTRICTS_FILE}'").fetchone()[0]
        print(f"Districts file has {count} rows.")
    except Exception as e:
        print(f"Error reading districts: {e}")

    print(f"Querying {STATS_FILE}...")
    try:
        count = con.execute(f"SELECT count(*) FROM '{STATS_FILE}'").fetchone()[0]
        print(f"Stats file has {count} rows.")
    except Exception as e:
        print(f"Error reading stats: {e}")

    print("Testing Join...")
    try:
        query = f"""
            SELECT 
                v.name, 
                COALESCE(s.green_area_m2, 0) as green_area_m2
            FROM '{DISTRICTS_FILE}' v 
            LEFT JOIN '{STATS_FILE}' s ON v.name = s.name
        """
        df = con.execute(query).fetchdf()
        print(f"Join returned {len(df)} rows.")
        print(df.head())
    except Exception as e:
        print(f"Error executing join: {e}")

if __name__ == "__main__":
    main()
