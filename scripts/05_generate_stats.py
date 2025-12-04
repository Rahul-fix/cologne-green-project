#!/usr/bin/env python3
import geopandas as gpd
import rasterio
import rasterio.features
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import box

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
STATS_DIR = DATA_DIR / "stats"
STATS_DIR.mkdir(exist_ok=True)

BOUNDARIES_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"

def calculate_ndvi(nir, red):
    # Avoid division by zero
    denominator = (nir + red)
    denominator[denominator == 0] = 0.0001
    ndvi = (nir - red) / denominator
    return ndvi

def calculate_stats(mask_path, boundaries):
    print(f"Calculating stats for {mask_path.name}...")
    
    # Find corresponding raw file
    # mask: name_mask.tif -> raw: name.jp2
    raw_name = mask_path.name.replace("_mask.tif", ".jp2").replace("_mask_COG.tif", ".jp2")
    raw_path = RAW_DIR / raw_name
    
    if not raw_path.exists():
        print(f"⚠️ Raw file {raw_name} not found. Calculating Area ONLY (skipping NDVI).")
        ndvi = None
    
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1)
        transform = src_mask.transform
        crs = src_mask.crs
        bounds = src_mask.bounds
        res = src_mask.res
        pixel_area = res[0] * res[1]
        
        if ndvi is None and raw_path.exists():
             # Read Raw and Calculate NDVI
            with rasterio.open(raw_path) as src_raw:
                # Read Red (1) and NIR (4)
                if src_raw.shape != src_mask.shape:
                    # print("⚠️ Dimension mismatch between mask and raw. Resampling raw...")
                    red = src_raw.read(1, out_shape=src_mask.shape, resampling=rasterio.enums.Resampling.nearest).astype(float)
                    nir = src_raw.read(4, out_shape=src_mask.shape, resampling=rasterio.enums.Resampling.nearest).astype(float)
                else:
                    red = src_raw.read(1).astype(float)
                    nir = src_raw.read(4).astype(float)
                    
                ndvi = calculate_ndvi(nir, red)

    # Reproject boundaries to match image CRS
    if boundaries.crs != crs:
        boundaries = boundaries.to_crs(crs)

    # Filter boundaries to those intersecting the image
    img_box = box(*bounds)
    intersecting_boundaries = boundaries[boundaries.intersects(img_box)].copy()
    
    if intersecting_boundaries.empty:
        print("No districts intersect this tile.")
        return None

    # Rasterize boundaries
    # We map each district name to a unique integer ID
    # Create a mapping: name -> id
    intersecting_boundaries['dist_id'] = range(1, len(intersecting_boundaries) + 1)
    id_to_name = intersecting_boundaries.set_index('dist_id')['name'].to_dict()
    
    # Generator of (geometry, id)
    shapes = ((geom, val) for geom, val in zip(intersecting_boundaries.geometry, intersecting_boundaries.dist_id))
    
    district_grid = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=rasterio.int32
    )

    # Calculate Stats
    # We want stats where mask == 1 (Green) AND district_grid > 0
    
    stats_list = []
    
    # Iterate over unique districts found in the grid
    unique_ids = np.unique(district_grid)
    unique_ids = unique_ids[unique_ids != 0] # Exclude background
    
    for dist_id in unique_ids:
        dist_name = id_to_name[dist_id]
        
        # Mask for this district and green area
        # Using boolean indexing
        valid_mask = (district_grid == dist_id) & (mask == 1)
        
        count = valid_mask.sum()
        
        if count > 0:
            area_m2 = count * pixel_area
            
            if ndvi is not None:
                ndvi_sum = ndvi[valid_mask].sum()
            else:
                ndvi_sum = 0.0
            
            stats_list.append({
                'name': dist_name,
                'green_area_m2': area_m2,
                'ndvi_sum': ndvi_sum,
                'ndvi_count': count if ndvi is not None else 0
            })
            
    if not stats_list:
        return None
        
    return pd.DataFrame(stats_list)

def main():
    if not BOUNDARIES_FILE.exists():
        print("❌ Boundaries not found. Run scripts/04_convert_boundaries.py first.")
        return

    boundaries = gpd.read_parquet(BOUNDARIES_FILE)
    
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    if not mask_files:
        print("❌ No mask files found. Run scripts/02_process_local.py first.")
        return

    print(f"Found {len(mask_files)} masks to process.")
    
    all_stats = []
    for mask_file in mask_files:
        try:
            stats = calculate_stats(mask_file, boundaries)
            if stats is not None:
                all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error processing {mask_file.name}: {e}")
            
    if all_stats:
        # Aggregate by district
        # Sum the areas, sums, and counts
        combined = pd.concat(all_stats)
        final_stats = combined.groupby('name').sum().reset_index()
        
        # Calculate weighted mean NDVI
        # Avoid division by zero
        final_stats['ndvi_mean'] = 0.0
        mask_valid = final_stats['ndvi_count'] > 0
        final_stats.loc[mask_valid, 'ndvi_mean'] = (
            final_stats.loc[mask_valid, 'ndvi_sum'] / final_stats.loc[mask_valid, 'ndvi_count']
        )
        
        # Drop temp columns if desired, or keep them
        # final_stats = final_stats.drop(columns=['ndvi_sum', 'ndvi_count'])
        
        output_path = STATS_DIR / "stats.parquet"
        final_stats.to_parquet(output_path)
        print(f"✅ Stats saved to {output_path}")
        print(final_stats[['name', 'green_area_m2', 'ndvi_mean']].head())
    else:
        print("No stats generated.")

if __name__ == "__main__":
    main()
