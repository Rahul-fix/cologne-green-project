#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path
import zipfile

# Configuration
DAOUTPUT_DIR = Path("data/boundaries")
BOUNDARIES = {
    "Stadtviertel.zip": "https://www.offenedaten-koeln.de/sites/default/files/Stadtviertel.zip",
    "Stadtbezirk.zip": "https://www.offenedaten-koeln.de/sites/default/files/Stadtbezirk.zip"
}

def download_file(url, local_path):
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Saved to {local_path}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    except Exception as e:
        print(f"❌ Error unzipping {zip_path}: {e}")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    
    for filename, url in BOUNDARIES.items():
        local_path = DATA_DIR / filename
        if not local_path.exists():
            download_file(url, local_path)
        else:
            print(f"File {filename} already exists. Skipping download.")
            
        # Unzip
        unzip_file(local_path, DATA_DIR)

if __name__ == "__main__":
    main()
