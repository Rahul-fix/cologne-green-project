#!/usr/bin/env python3
import geopandas as gpd
import rasterio
import rasterio.features
import pandas as pd
import numpy as np
import os
from pathlib import Path
from shapely.geometry import box

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
STATS_DIR = DATA_DIR / "stats"
STATS_DIR.mkdir(exist_ok=True)

BOUNDARIES_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
OUTPUT_FILE = STATS_DIR / "extended_stats.parquet"

CLASS_LABELS = {
    1: 'Building', 2: 'Impervious', 3: 'Barren', 4: 'Grass', 5: 'Brush',
    6: 'Agriculture', 7: 'Tree', 8: 'Water', 9: 'Herbaceous', 10: 'Shrub',
    11: 'Moss', 12: 'Lichen', 13: 'Unknown'
}

def calculate_ndvi(nir, red):
    denominator = (nir + red)
    denominator[denominator == 0] = 0.0001
    ndvi = (nir - red) / denominator
    return ndvi

    # ... [Imports]
import concurrent.futures
import multiprocessing

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
STATS_DIR = DATA_DIR / "stats"
STATS_DIR.mkdir(exist_ok=True)

BOUNDARIES_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
OUTPUT_FILE = STATS_DIR / "extended_stats.parquet"

def process_tile_wrapper(args):
    # Wrapper to unpack arguments for pool map
    tile_name, boundaries = args
    try:
        return process_tile(tile_name, boundaries)
    except Exception as e:
        return None

def process_tile(tile_id, boundaries):
    # Moved the logic from the loop into this function
    # Define paths
    raw_path = RAW_DIR / f"{tile_id}.jp2"
    mask_path = PROCESSED_DIR / f"{tile_id}_mask.tif"
    
    # Determine reference file
    ref_path = mask_path if mask_path.exists() else raw_path
    if not ref_path.exists(): return None
    
    stats_list = []
    try:
        # Open Reference to get grid info
        with rasterio.open(ref_path) as src:
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            res = src.res
            shape = src.shape
            pixel_area = res[0] * res[1]
        
        # Load NDVI if raw exists
        ndvi = None
        if raw_path.exists():
            try:
                with rasterio.open(raw_path) as src_raw:
                    if src_raw.count >= 4:
                        red = src_raw.read(1).astype(float)
                        nir = src_raw.read(4).astype(float)
                        ndvi = calculate_ndvi(nir, red)
            except: pass

        # Load Mask if exists
        mask = None
        if mask_path.exists():
            try:
                with rasterio.open(mask_path) as src_mask:
                    mask = src_mask.read(1)
            except: pass
        
        curr_boundaries = boundaries # Assume passed in correct CRS or we check
        if curr_boundaries.crs != crs:
            try: curr_boundaries = curr_boundaries.to_crs(crs)
            except: 
                 try: curr_boundaries = curr_boundaries.to_crs("EPSG:25832")
                 except: return None

        img_box = box(*bounds)
        intersecting = curr_boundaries[curr_boundaries.intersects(img_box)].copy()
        
        if intersecting.empty: return None

        # Rasterize Districts
        intersecting['dist_id'] = range(1, len(intersecting) + 1)
        id_to_name = intersecting.set_index('dist_id')['name'].to_dict()
        
        shapes = ((geom, val) for geom, val in zip(intersecting.geometry, intersecting['dist_id']))
        
        district_grid = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=rasterio.int32
        )
        
        unique_ids_in_tile = np.unique(district_grid)
        unique_ids_in_tile = unique_ids_in_tile[unique_ids_in_tile != 0]
        
        for dist_id in unique_ids_in_tile:
            dist_name = id_to_name[dist_id]
            dist_mask = (district_grid == dist_id)
            
            # 1. Class Areas
            class_areas = {}
            if mask is not None:
                 for cls_id, cls_name in CLASS_LABELS.items():
                    count = np.sum((mask == cls_id) & dist_mask)
                    area = count * pixel_area
                    class_areas[f"area_{cls_id}"] = area
            else:
                for cls_id in CLASS_LABELS: class_areas[f"area_{cls_id}"] = 0.0

            # 2. NDVI Stats
            n_sum = 0.0
            n_count = 0
            if ndvi is not None:
                val_pixels = dist_mask & (~np.isnan(ndvi))
                if np.any(val_pixels):
                    n_sum = np.nansum(ndvi[val_pixels])
                    n_count = np.count_nonzero(val_pixels)
            
            record = {
                'name': dist_name,
                'ndvi_sum': n_sum,
                'ndvi_count': n_count,
                **class_areas
            }
            stats_list.append(record)
    except Exception as e:
        print(f"❌ Error processing {tile_id}: {e}")
        return None
        
    return stats_list

def main():
    if not BOUNDARIES_FILE.exists():
        print("❌ Boundaries not found.")
        return

    boundaries = gpd.read_parquet(BOUNDARIES_FILE)
    
    # Identify Tiles
    raw_files = list(RAW_DIR.glob("*.jp2"))
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    
    tile_ids = set()
    for p in raw_files: tile_ids.add(p.stem)
    for p in mask_files: tile_ids.add(p.name.replace("_mask.tif", "").replace("_mask", ""))
    
    sorted_tiles = sorted(list(tile_ids))
    print(f"Found {len(sorted_tiles)} total tiles (Raw + Processed).")
    
    try:
        boundaries = boundaries.to_crs("EPSG:25832")
    except: pass

    all_stats = []
    
    max_workers = min(12, os.cpu_count() or 4)
    print(f"🚀 Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {executor.submit(process_tile, tile_id, boundaries): tile_id for tile_id in sorted_tiles}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_tile):
            tile_id = future_to_tile[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    all_stats.extend(result)
                if completed % 10 == 0:
                    print(f"[{completed}/{len(sorted_tiles)}] processed...")
            except Exception as e:
                print(f"❌ Worker error for {tile_id}: {e}")

    if all_stats:
        combined = pd.concat([pd.DataFrame(s, index=[0]) if isinstance(s, dict) else pd.DataFrame(s) for s in all_stats], ignore_index=True)
        combined = pd.DataFrame(all_stats)
        
        final_stats = combined.groupby('name').sum().reset_index()
        
        final_stats['ndvi_mean'] = 0.0
        mask_valid = final_stats['ndvi_count'] > 0
        final_stats.loc[mask_valid, 'ndvi_mean'] = (
            final_stats.loc[mask_valid, 'ndvi_sum'] / final_stats.loc[mask_valid, 'ndvi_count']
        )
        
        green_cols = [f"area_{c}" for c in [4, 5, 6, 7, 9, 10, 11, 12]]
        existing_cols = [c for c in green_cols if c in final_stats.columns]
        final_stats['green_area_m2'] = final_stats[existing_cols].sum(axis=1) if existing_cols else 0.0

        print(f"Saving to {OUTPUT_FILE}...")
        final_stats.to_parquet(OUTPUT_FILE)
        print("✅ Done.")
    else:
        print("No stats generated.")

if __name__ == "__main__":
    main()

