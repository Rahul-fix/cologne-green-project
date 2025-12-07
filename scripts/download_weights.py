#!/usr/bin/env python3
"""
Download FLAIR-HUB model weights from Hugging Face Hub.
"""
from huggingface_hub import snapshot_download
from pathlib import Path

# Target directory for weights
# We place it inside DL_cologne_green to keep things organized
TARGET_DIR = Path("DL_cologne_green/FLAIR-HUB_LC-A_IR_swinbase-upernet")

def main():
    print(f"⬇️  Downloading model weights to {TARGET_DIR}...")
    
    try:
        snapshot_download(
            repo_id="IGNF/FLAIR-HUB_LC-A_IR_swinbase-upernet",
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()
