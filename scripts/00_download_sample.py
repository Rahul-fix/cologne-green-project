#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path

# Configuration
# Tile covering Cologne Cathedral area (approx)
# Based on UTM 32N coordinates ~356000, 5645000
TILE_NAME = "dop10rgbi_32_356_5645_1_nw_2025" 
BASE_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/"
DAOUTPUT_DIR = Path("data/raw")

def download_file(url, local_path):
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Saved to {local_path}")
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        raise Exception(f"Download error: {e}")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    
    filename = f"{TILE_NAME}.jp2"
    url = f"{BASE_URL}{filename}"
    local_path = DATA_DIR / filename
    
    if local_path.exists():
        print(f"File {filename} already exists. Skipping.")
    else:
        try:
            download_file(url, local_path)
            print("✅ Download successful!")
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            # Try 2024 if 2025 fails (metadata said 2025 but sometimes files lag)
            try:
                fallback_name = TILE_NAME.replace("2025", "2024")
                print(f"Retrying with 2024 version: {fallback_name}")
                url = f"{BASE_URL}{fallback_name}.jp2"
                local_path = DATA_DIR / f"{fallback_name}.jp2"
                download_file(url, local_path)
                print("✅ Download successful (2024 version)!")
            except Exception as e2:
                 print(f"❌ Error downloading fallback: {e2}")

if __name__ == "__main__":
    main()
