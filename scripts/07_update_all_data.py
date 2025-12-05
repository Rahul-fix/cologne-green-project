#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_step(command, description):
    print(f"\n--- {description} ---")
    try:
        subprocess.run(command, check=True)
        print("✅ Success")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

def main():
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"

    print("🚀 Starting Data Update Pipeline...")

    # Step 1: Download Processed Data & Boundaries from HF
    # Only download 'processed' and 'boundaries' to save bandwidth/API limits on 'raw'
    hf_download_script = scripts_dir / "download_from_hf.py"
    run_step(
        ["python", str(hf_download_script), "--folders", "processed", "boundaries"],
        "Step 1: Downloading Processed Data from Hugging Face"
    )

    # Step 2: Download Raw Data from OpenNRW
    # Uses download_all_tiles.py to fetch JP2 files directly
    raw_download_script = scripts_dir / "download_all_tiles.py"
    run_step(
        ["python", str(raw_download_script), "--workers", "8"],
        "Step 2: Downloading Raw Satellite Images (OpenNRW)"
    )

    # Step 2: Calculate Stats
    # Generates extended_stats.parquet
    stats_script = scripts_dir / "06_calculate_full_stats.py"
    run_step(
        ["python", str(stats_script)],
        "Step 2: Calculating Full Statistics"
    )

    print("\n✨ Pipeline Complete! App data is ready.")

if __name__ == "__main__":
    main()
