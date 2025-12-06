#!/usr/bin/env python3
"""
Create web-optimized versions of satellite tiles for fast visualization.
Downsamples raw JP2 imagery and masks to ~1-2m resolution.
Pre-calculates NDVI for instant display.
"""
import rasterio
from rasterio.enums import Resampling
import os
from pathlib import Path
import concurrent.futures
import numpy as np

# Configuration
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OPTIMIZED_DIR = DATA_DIR / "web_optimized"
OPTIMIZED_DIR.mkdir(exist_ok=True)

DOWNSCALE_FACTOR = 10  # 10000x10000 -> 1000x1000
JPEG_QUALITY = 75

def get_optimized_profile(src_profile, height, width, transform, count=3, compress='jpeg'):
    profile = src_profile.copy()
    profile.update({
        'driver': 'GTiff',
        'count': count,
        'height': height,
        'width': width,
        'transform': transform,
        'compress': compress,
        'tiled': True,
    })
    if compress == 'jpeg':
        profile.update({'jpeg_quality': JPEG_QUALITY, 'photometric': 'YCBCR'})
    else:
        # For masks/NDVI (LZW or Deflate)
        if 'jpeg_quality' in profile: del profile['jpeg_quality']
        if 'photometric' in profile: del profile['photometric']
    return profile

def optimize_rgb(jp2_path):
    try:
        filename = jp2_path.stem
        output_path = OPTIMIZED_DIR / f"{filename}.tif"
        if output_path.exists(): return f"⏩ RGB exists: {filename}"

        with rasterio.open(jp2_path) as src:
            # New dims
            h, w = src.height // DOWNSCALE_FACTOR, src.width // DOWNSCALE_FACTOR
            transform = src.transform * src.transform.scale(src.width/w, src.height/h)
            
            # Read RGB (1,2,3)
            data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
            
            profile = get_optimized_profile(src.profile, h, w, transform, count=3, compress='jpeg')
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
        return f"✅ Optimized RGB: {filename}"
    except Exception as e: return f"❌ Failed RGB {jp2_path.name}: {e}"

def optimize_mask(mask_path):
    try:
        filename = mask_path.stem # e.g. tile_mask
        # remove _mask suffix for consistency? No, keep it as {tile}_mask.tif for clarity
        output_path = OPTIMIZED_DIR / f"{filename}.tif"
        if output_path.exists(): return f"⏩ Mask exists: {filename}"

        with rasterio.open(mask_path) as src:
            h, w = src.height // DOWNSCALE_FACTOR, src.width // DOWNSCALE_FACTOR
            transform = src.transform * src.transform.scale(src.width/w, src.height/h)
            
            # Read Mask - NEAREST NEIGHBOR to preserve integer classes!
            data = src.read(1, out_shape=(1, h, w), resampling=Resampling.nearest)
            
            # Use LZW or Deflate for masks (lossless, handles integers well)
            profile = get_optimized_profile(src.profile, h, w, transform, count=1, compress='deflate')
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
        return f"✅ Optimized Mask: {filename}"
    except Exception as e: return f"❌ Failed Mask {mask_path.name}: {e}"

def optimize_ndvi(jp2_path):
    try:
        filename = jp2_path.stem
        output_path = OPTIMIZED_DIR / f"{filename}_ndvi.tif"
        if output_path.exists(): return f"⏩ NDVI exists: {filename}"

        with rasterio.open(jp2_path) as src:
            # Need 4 bands for NDVI
            if src.count < 4: return f"⚠️ No NIR band for {filename}"

            h, w = src.height // DOWNSCALE_FACTOR, src.width // DOWNSCALE_FACTOR
            transform = src.transform * src.transform.scale(src.width/w, src.height/h)

            # Read Red (1) and NIR (4). Note: Cologne data usually R=1, G=2, B=3, NIR=4
            # Read downsampled directly using 'average' resampling
            r = src.read(1, out_shape=(1, h, w), resampling=Resampling.average).astype('float32')
            nir = src.read(4, out_shape=(1, h, w), resampling=Resampling.average).astype('float32')
            
            # Calculate NDVI
            ndvi = (nir - r) / (nir + r + 1e-8)
            
            # Quantize for display? 
            # Folium/Leaflet can't handle float32 raw properly without custom colormap on client?
            # Actually, app uses matplotlib to colorize.
            # Keeping as float32 is better for accurate coloring, but size is bigger.
            # 1000x1000 float32 is ~4MB uncompressed. Deflate compresses it well if smooth.
            # Let's keep typical NDVI range -1 to 1.
            
            profile = get_optimized_profile(src.profile, h, w, transform, count=1, compress='deflate')
            profile.update({'dtype': 'float32', 'nodata': None})
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(ndvi, 1)
                
        return f"✅ Generated NDVI: {filename}"
    except Exception as e: return f"❌ Failed NDVI {jp2_path.name}: {e}"

def process_tile_group(jp2_path):
    res = []
    # 1. RGB
    res.append(optimize_rgb(jp2_path))
    
    # 2. NDVI (derived from raw)
    res.append(optimize_ndvi(jp2_path))
    
    # 3. Mask (if exists in processed)
    mask_path = PROCESSED_DIR / f"{jp2_path.stem}_mask.tif"
    if mask_path.exists():
        res.append(optimize_mask(mask_path))
        
    return res

def main():
    if not RAW_DIR.exists():
        print(f"❌ Error: {RAW_DIR} not found.")
        return

    jp2_files = list(RAW_DIR.glob("*.jp2"))
    total_files = len(jp2_files)
    
    print(f"found {total_files} raw tiles.")
    print(f"Target Directory: {OPTIMIZED_DIR}")
    
    max_workers = min(12, os.cpu_count() or 4)
    print(f"🚀 Starting parallel optimization (RGB, Mask, NDVI) with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tile_group, p): p for p in jp2_files}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            results = future.result()
            # results is a list of strings
            # only print errors or progress
            if completed % 10 == 0:
                print(f"[{completed}/{total_files}] Processed tile group.")
            for r in results:
                if "Failed" in r or "Error" in r:
                    print(r)

    print("✅ All optimization tasks finished.")

if __name__ == "__main__":
    main()
