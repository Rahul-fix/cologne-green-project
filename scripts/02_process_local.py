#!/usr/bin/env python3
import os
import sys
import yaml
import time
import torch
import rasterio
import argparse
import traceback
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import FLAIR-HUB components
from flair_zonal_detection.config import load_config, validate_config
from flair_zonal_detection.model_utils import build_inference_model, compute_patch_sizes
from flair_zonal_detection.inference import (
    initialize_geometry_and_resolutions,
    prep_dataset,
    init_outputs,
    inference_and_write,
    postpro_outputs
)
from flair_zonal_detection.slicing import generate_patches_from_reference

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_CONFIG_PATH = Path("DL_cologne_green/config_vm_inference.yaml")

def main():
    parser = argparse.ArgumentParser(description="Run optimized inference on local JP2 tiles.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to inference config file")
    args = parser.parse_args()

    # 1. Find images
    jp2_files = list((DATA_DIR / "raw").glob("*.jp2"))
    if not jp2_files:
        print("❌ No .jp2 files found in data/raw/.")
        return
    
    # Filter out already processed files
    images_to_process = []
    for img_path in jp2_files:
        mask_path = PROCESSED_DIR / f"{img_path.stem}_mask.tif"
        if mask_path.exists():
            print(f"✅ Output {mask_path.name} already exists. Skipping.")
        else:
            images_to_process.append(img_path)
            
    if not images_to_process:
        print("✅ All images already processed!")
        return

    print(f"Found {len(images_to_process)} images to process.")

    # 2. Load Config & Model (ONCE)
    print("\n[1/3] Loading Configuration and Model...")
    try:
        # Load base config
        config = load_config(str(args.config))
        
        # Fix weights path if needed
        original_weights = config.get('model_weights', '')
        if original_weights.startswith('./FLAIR-HUB'):
             config['model_weights'] = str(Path("DL_cologne_green") / original_weights.lstrip('./'))
        
        # Set device
        config['device'] = torch.device("cuda" if config.get("use_gpu", torch.cuda.is_available()) else "cpu")
        print(f"Using device: {config['device']}")
        
        # Compute patch sizes (independent of image size)
        patch_sizes = compute_patch_sizes(config)
        
        # Build Model
        start_model = time.time()
        model = build_inference_model(config, patch_sizes).to(config['device'])
        model.eval() # Ensure eval mode
        print(f"✅ Loaded model in {time.time() - start_model:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        traceback.print_exc()
        return

    # 3. Process Images Loop
    print("\n[2/3] Starting Inference Loop...")
    
    for i, img_path in enumerate(images_to_process):
        print(f"\n--- Processing [{i+1}/{len(images_to_process)}] {img_path.name} ---")
        
        try:
            # Create a per-image config copy
            # We need a deep copy of specific dicts if we modify them, 
            # but standard dict copy might be enough if we only change top-level or specific keys.
            # Safer to reload or copy carefully. 
            # Since `initialize_geometry_and_resolutions` modifies config in-place, 
            # we should probably reload the base config or be very careful.
            # Actually, `load_config` is cheap (just reading YAML). 
            # But we want to keep the MODEL loaded.
            
            # Let's use the base config structure but update it.
            # `initialize_geometry_and_resolutions` sets 'image_bounds', 'modality_resolutions', etc.
            # These are specific to the image.
            
            current_config = config.copy()
            # Deep copy modalities to avoid polluting base config
            import copy
            current_config['modalities'] = copy.deepcopy(config['modalities'])
            
            # Update Input Path
            if 'AERIAL_RGBI' in current_config['modalities']:
                current_config['modalities']['AERIAL_RGBI']['input_img_path'] = str(img_path.absolute())
            else:
                print("❌ 'AERIAL_RGBI' modality missing in config.")
                continue

            # Update Output Path
            current_config['output_path'] = str(PROCESSED_DIR.absolute())
            current_config['output_name'] = img_path.stem
            
            # Initialize Geometry (Per Image)
            current_config = initialize_geometry_and_resolutions(current_config)
            
            # Generate Patches
            tiles_gdf = generate_patches_from_reference(current_config)
            # print(f"Sliced into {len(tiles_gdf)} tiles")
            
            # Prepare Dataset & DataLoader
            dataset = prep_dataset(current_config, tiles_gdf, patch_sizes)
            dataloader = DataLoader(
                dataset, 
                batch_size=current_config['batch_size'], 
                num_workers=current_config['num_worker']
            )
            
            # Init Outputs
            # We need to open the reference image to get profile
            with rasterio.open(current_config['modalities'][current_config['reference_modality']]['input_img_path']) as ref_img:
                output_files, temp_paths = init_outputs(current_config, ref_img)
                
                # Run Inference
                inference_and_write(model, dataloader, tiles_gdf, current_config, output_files, ref_img)
            
            # Post-processing (COG)
            postpro_outputs(temp_paths, current_config)
            
            # Rename/Cleanup to match expected output format
            # The script logic expects {stem}_mask.tif
            # FLAIR-HUB produces {stem}_AERIAL_LABEL-COSIA_argmax.tif
            
            expected_output = PROCESSED_DIR / f"{img_path.stem}_mask.tif"
            
            # Find the generated file
            candidates = list(PROCESSED_DIR.glob(f"{img_path.stem}*argmax*.tif"))
            source_file = None
            if candidates:
                source_file = candidates[0] # Take the first one (usually only one task)
            
            if source_file and source_file.exists():
                # Enforce EPSG:25832 and rename
                with rasterio.open(source_file) as src:
                    data = src.read()
                    profile = src.profile.copy()
                    profile.update(crs=rasterio.crs.CRS.from_epsg(25832), driver='GTiff')
                    
                    with rasterio.open(expected_output, 'w', **profile) as dst:
                        dst.write(data)
                
                # Remove original
                os.remove(source_file)
                print(f"✅ Saved final mask: {expected_output.name}")
            else:
                print(f"⚠️ Could not find output file for {img_path.name}")

        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")
            traceback.print_exc()
            # Continue to next image!

    print("\n✅ All processing finished.")

if __name__ == "__main__":
    main()
