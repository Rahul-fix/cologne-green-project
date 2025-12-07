#!/usr/bin/env python3
"""
Sync processed data to Google Drive using rclone.
Usage: python scripts/sync_to_drive.py <remote_path>
Example: python scripts/sync_to_drive.py gdrive:cologne-green-data
"""
import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data")

def run_rclone(source, dest, description):
    print(f"\n🚀 Syncing {description}...")
    # -P shows progress, --transfers=8 for parallel uploads
    cmd = ["rclone", "sync", str(source), str(dest), "-P", "--transfers", "8"]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to sync {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Sync data to Google Drive via rclone.")
    parser.add_argument("remote_path", help="Rclone remote path (e.g., gdrive:project-folder)")
    args = parser.parse_args()

    remote_base = args.remote_path.rstrip("/")

    # 1. Sync Processed Masks
    run_rclone(
        DATA_DIR / "processed", 
        f"{remote_base}/processed", 
        "Processed Data (Masks & NDVI)"
    )

    # 2. Sync Statistics
    run_rclone(
        DATA_DIR / "stats", 
        f"{remote_base}/stats", 
        "Statistics"
    )

    # 3. Sync Boundaries (Optional, good for backup)
    run_rclone(
        DATA_DIR / "boundaries", 
        f"{remote_base}/boundaries", 
        "Boundaries"
    )

    print("\n✨ All data synced to Google Drive!")

if __name__ == "__main__":
    main()
