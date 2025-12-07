#!/usr/bin/env python3
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw")

def main():
    jp2_files = list(DATA_DIR.glob("*.jp2"))
    
    if not jp2_files:
        print("❌ No .jp2 files found in data/ directory.")
        return

    # Pick the first one
    img_path = jp2_files[0]
    print(f"🔍 Inspecting {img_path}...")

    try:
        with rasterio.open(img_path) as src:
            print(f"✅ Driver: {src.driver}")
            print(f"✅ Width: {src.width}, Height: {src.height}")
            print(f"✅ CRS: {src.crs}")
            print(f"✅ Bounds: {src.bounds}")
            print(f"✅ Count (bands): {src.count}")
            
            # Check if CRS matches expected EPSG:25832 (ETRS89 / UTM zone 32N)
            if src.crs and src.crs.to_epsg() == 25832:
                print("✅ CRS matches expected EPSG:25832")
            else:
                print(f"⚠️ Unexpected CRS: {src.crs}")

            # Create a thumbnail
            print("🖼️ Generating thumbnail...")
            plt.figure(figsize=(10, 10))
            # Read first 3 bands (RGB)
            if src.count >= 3:
                # Read a low-res overview for speed
                # factor=10 means 1/10th resolution
                data = src.read([1, 2, 3], out_shape=(3, int(src.height / 10), int(src.width / 10)))
                show(data, transform=src.transform)
                plt.title(f"Thumbnail: {img_path.name}")
                output_thumb = DATA_DIR / f"{img_path.stem}_thumb.png"
                plt.savefig(output_thumb)
                print(f"✅ Thumbnail saved to {output_thumb}")
            else:
                print("⚠️ Image has fewer than 3 bands, skipping RGB thumbnail.")

    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
