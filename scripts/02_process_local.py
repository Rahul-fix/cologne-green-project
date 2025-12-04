#!/usr/bin/env python3
import os
import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
import yaml
import subprocess
import shutil

import argparse

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Default path
DEFAULT_CONFIG_PATH = Path("DL_cologne_green/config_nrw_inference.yaml")

def calculate_ndvi(nir, red):
    # Avoid division by zero
    denominator = (nir + red)
    denominator[denominator == 0] = 0.0001
    ndvi = (nir - red) / denominator
    return ndvi

def process_image(img_path, config_path):
    print(f"Processing {img_path}...")
    
    # 1. Calculate NDVI (Still useful for quick visualization)
    with rasterio.open(img_path) as src:
        # Check band count. NRW DOP usually has 4 bands: R, G, B, NIR (or similar)
        # config_nrw_inference.yaml says: channels: [4, 1, 2] # NIR, R, G
        # So band 4 is NIR, band 1 is Red.
        
        if src.count < 4:
            print(f"⚠️ Image has {src.count} bands, expected at least 4 for NDVI (NIR). Skipping NDVI.")
        else:
            red = src.read(1).astype(float)
            nir = src.read(4).astype(float)
            
            # NDVI Calculation (On-Demand in App now)
            # print("Calculating NDVI...")
            # ndvi = calculate_ndvi(nir, red)
            
            # # Save NDVI with compression (Int16 scaled by 10000)
            # # NDVI is -1 to 1. Scaled: -10000 to 10000.
            # # This reduces size significantly compared to Float32.
            # ndvi_path = PROCESSED_DIR / f"{img_path.stem}_ndvi.tif"
            
            # # Scale and cast to int16
            # ndvi_int16 = (ndvi * 10000).astype(np.int16)
            
            # profile = src.profile
            # profile.update(
            #     dtype=rasterio.int16, 
            #     count=1, 
            #     driver='GTiff',
            #     compress='lzw',
            #     predictor=2,
            #     nodata=-32768 # Optional nodata value
            # )
            
            # with rasterio.open(ndvi_path, 'w', **profile) as dst:
            #     dst.write(ndvi_int16, 1)
            #     # Add metadata about scaling
            #     dst.update_tags(scale=0.0001, offset=0)
                
            # print(f"✅ NDVI saved to {ndvi_path} (Int16 Compressed)")

    # 2. Run FLAIR-HUB Inference
    target_output = PROCESSED_DIR / f"{img_path.stem}_mask.tif"
    if target_output.exists():
        print(f"✅ Mask {target_output.name} already exists. Skipping inference.")
        return

    print("Running FLAIR-HUB Segmentation...")
    
    if not config_path.exists():
        print(f"❌ Config not found at {config_path}. Cannot run inference.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config for this specific image
    # We need to point 'AERIAL_RGBI' to our local image
    if 'AERIAL_RGBI' in config['modalities']:
        config['modalities']['AERIAL_RGBI']['input_img_path'] = str(img_path.absolute())
    else:
        print("❌ 'AERIAL_RGBI' modality missing in config.")
        return

    # Update output path
    config['output_path'] = str(PROCESSED_DIR.absolute())
    config['output_name'] = img_path.stem # Output will be {stem}_pred.tif usually
    
    # Ensure model weights path is correct relative to where we run
    # Config has: ./FLAIR-HUB_LC-A_IR_swinbase-upernet/...
    # We are running from project root, so DL_cologne_green/...
    original_weights = config.get('model_weights', '')
    if original_weights.startswith('./FLAIR-HUB'):
         config['model_weights'] = str(Path("DL_cologne_green") / original_weights.lstrip('./'))
    
    # Save temp config
    temp_config_path = PROCESSED_DIR / f"config_{img_path.stem}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Run inference command
    # Using python -m to avoid PATH issues
    import sys
    cmd = [sys.executable, "-m", "flair_zonal_detection.main", "--config", str(temp_config_path)]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Inference complete for {img_path.name}")
        
        # Rename output to standard mask name
        # FLAIR-HUB output naming can vary. It seems to be:
        # {stem}_AERIAL_LABEL-COSIA_argmax_COG.tif (if COG enabled)
        # {stem}_AERIAL_LABEL-COSIA_argmax.tif (if COG disabled)
        
        target_output = PROCESSED_DIR / f"{img_path.stem}_mask.tif"
        
        # Search for likely outputs
        candidates = list(PROCESSED_DIR.glob(f"{img_path.stem}*argmax*.tif"))
        
        # Prefer COG if available
        cog_candidates = [c for c in candidates if "COG" in c.name]
        
        source_file = None
        if cog_candidates:
            source_file = cog_candidates[0]
        elif candidates:
            source_file = candidates[0]
            
        if source_file and source_file.exists():
            print(f"✅ Found output: {source_file.name}")
            
            # Rewrite with correct CRS (EPSG:25832) to avoid EngineeringCRS issues
            # We read the source and write to target_output with explicit CRS
            try:
                with rasterio.open(source_file) as src:
                    data = src.read()
                    profile = src.profile.copy()
                    
                    # Force EPSG:25832
                    profile.update(
                        crs=rasterio.crs.CRS.from_epsg(25832),
                        driver='GTiff' # Ensure GTiff
                    )
                    
                    with rasterio.open(target_output, 'w', **profile) as dst:
                        dst.write(data)
                        
                print(f"✅ Saved {target_output.name} with enforced EPSG:25832")
                
                # Remove original source file
                os.remove(source_file)
                
            except Exception as e:
                print(f"❌ Failed to rewrite with correct CRS: {e}")
                # Fallback: just move it if rewrite fails
                if not target_output.exists():
                    shutil.move(source_file, target_output)

            # Clean up other artifacts
            for c in candidates:
                if c != source_file and c.exists():
                    os.remove(c)
        else:
            print(f"⚠️ Could not find expected output file for {img_path.stem}")
            print(f"   Found candidates: {[c.name for c in candidates]}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed: {e}")
    finally:
        if temp_config_path.exists():
            os.remove(temp_config_path)

def main():
    parser = argparse.ArgumentParser(description="Run inference on local JP2 tiles.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to inference config file")
    args = parser.parse_args()

    # Look in data/raw/
    jp2_files = list((DATA_DIR / "raw").glob("*.jp2"))
    
    if not jp2_files:
        print("❌ No .jp2 files found in data/raw/.")
        return

    print(f"Found {len(jp2_files)} images to process.")

    for i, img_path in enumerate(jp2_files):
        print(f"\n--- Processing [{i+1}/{len(jp2_files)}] {img_path.name} ---")
        
        # Check if output already exists
        mask_path = PROCESSED_DIR / f"{img_path.stem}_mask.tif"
        ndvi_path = PROCESSED_DIR / f"{img_path.stem}_ndvi.tif"
        
        if mask_path.exists():
            print(f"✅ Output {mask_path.name} already exists. Skipping.")
            continue
        
        try:
            process_image(img_path, args.config)
        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")

if __name__ == "__main__":
    main()
