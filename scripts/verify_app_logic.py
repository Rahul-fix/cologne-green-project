from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

def main():
    print("--- Verifying App Logic ---")
    
    # 1. Find processed files (masks)
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    print(f"Found {len(mask_files)} mask files in {PROCESSED_DIR}")
    
    available_tiles = set(f.stem.replace("_mask", "") for f in mask_files)
    print(f"Unique processed tiles: {len(available_tiles)}")
    
    # 2. Find raw files
    raw_files = list(RAW_DIR.glob("*.jp2"))
    print(f"Found {len(raw_files)} raw files in {RAW_DIR}")
    
    available_raw = set(f.stem for f in raw_files)
    
    # 3. Union (Logic from app_local.py)
    all_available = available_tiles.union(available_raw)
    print(f"Total unique available tiles (Union): {len(all_available)}")
    
    # 4. Simulate "All" selection
    tile_options = sorted(list(all_available))
    
    print("\n--- Tile Options (What the user sees) ---")
    for t in tile_options:
        status = []
        if t in available_tiles: status.append("Mask/NDVI")
        if t in available_raw: status.append("Raw")
        print(f"{t} [{', '.join(status)}]")
        
    print(f"\nTotal Options: {len(tile_options)}")

if __name__ == "__main__":
    main()
