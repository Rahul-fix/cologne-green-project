#!/usr/bin/env python3
"""
Download all satellite tiles listed in data/metadata/cologne_tiles.csv.
Uses parallel downloads for efficiency.
"""
import pandas as pd
from pathlib import Path
import urllib.request
import time
import concurrent.futures
import argparse
import os

DATA_DIR = Path("data")
TILES_CSV = DATA_DIR / "metadata" / "cologne_tiles.csv"
DOWNLOAD_DIR = DATA_DIR / "raw"

def download_tile(row):
    url = row['url']
    filename = row['filename']
    filepath = DOWNLOAD_DIR / filename
    
    if filepath.exists():
        return f"✅ {filename} exists"
        
    try:
        # User-Agent to avoid some server blocks
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
        )
        
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
            
        return f"⬇️  Downloaded {filename}"
    except Exception as e:
        return f"❌ Failed {filename}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Download satellite tiles.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    args = parser.parse_args()

    if not TILES_CSV.exists():
        print(f"❌ Error: {TILES_CSV} not found. Run scripts/find_cologne_tiles.py first.")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(TILES_CSV)
    total_tiles = len(df)
    print(f"📥 Found {total_tiles} tiles to download. Using {args.workers} workers.")
    
    # Convert dataframe to list of dicts for iteration
    rows = df.to_dict('records')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(download_tile, row) for row in rows]
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            # Print progress every 10 tiles or on error
            if "Failed" in result or completed % 10 == 0 or completed == total_tiles:
                print(f"[{completed}/{total_tiles}] {result}")

    print("✅ All downloads finished.")

if __name__ == "__main__":
    main()
