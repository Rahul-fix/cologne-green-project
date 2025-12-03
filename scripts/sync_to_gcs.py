#!/usr/bin/env python3
"""
Sync processed data to Google Cloud Storage (GCS).
Organizes flat local output into structured GCS folders:
- data/processed/*_mask.tif -> gs://BUCKET/processed/masks/
- data/processed/*_ndvi.tif -> gs://BUCKET/processed/ndvi/
- data/stats/*              -> gs://BUCKET/stats/
- data/boundaries/*         -> gs://BUCKET/boundaries/
"""

import argparse
import subprocess
from pathlib import Path
import shutil

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
STATS_DIR = DATA_DIR / "stats"
BOUNDARIES_DIR = DATA_DIR / "boundaries"

def run_gsutil(cmd):
    """Run a gsutil command."""
    full_cmd = ["gsutil"] + cmd
    print(f"Running: {' '.join(full_cmd)}")
    try:
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Sync data to GCS.")
    parser.add_argument("bucket", help="GCS Bucket name (e.g., cologne-green-project)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    bucket_uri = f"gs://{args.bucket}"
    if bucket_uri.endswith("/"):
        bucket_uri = bucket_uri[:-1]

    print(f"🚀 Syncing data to {bucket_uri}...")

    # 1. Sync Masks
    print("\n--- Syncing Masks ---")
    # We use -m for parallel upload
    # We filter for *_mask.tif
    # Since gsutil cp doesn't support complex regex include/exclude easily in one go for flat->nested,
    # we might need to be creative. 
    # Actually, simplest is to upload all *_mask.tif to processed/masks/
    
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    if mask_files:
        print(f"Found {len(mask_files)} mask files.")
        cmd = ["-m", "cp", str(PROCESSED_DIR / "*_mask.tif"), f"{bucket_uri}/processed/masks/"]
        if not args.dry_run:
            run_gsutil(cmd)
    else:
        print("No mask files found.")

    # 2. Sync NDVI
    print("\n--- Syncing NDVI ---")
    ndvi_files = list(PROCESSED_DIR.glob("*_ndvi.tif"))
    if ndvi_files:
        print(f"Found {len(ndvi_files)} NDVI files.")
        cmd = ["-m", "cp", str(PROCESSED_DIR / "*_ndvi.tif"), f"{bucket_uri}/processed/ndvi/"]
        if not args.dry_run:
            run_gsutil(cmd)
    else:
        print("No NDVI files found.")

    # 3. Sync Stats
    print("\n--- Syncing Stats ---")
    if STATS_DIR.exists():
        cmd = ["-m", "rsync", "-r", str(STATS_DIR), f"{bucket_uri}/stats/"]
        if not args.dry_run:
            run_gsutil(cmd)
    else:
        print("Stats directory not found.")

    # 4. Sync Boundaries
    print("\n--- Syncing Boundaries ---")
    if BOUNDARIES_DIR.exists():
        cmd = ["-m", "rsync", "-r", str(BOUNDARIES_DIR), f"{bucket_uri}/boundaries/"]
        if not args.dry_run:
            run_gsutil(cmd)
    else:
        print("Boundaries directory not found.")

    print("\n✅ Sync complete!")

if __name__ == "__main__":
    main()
