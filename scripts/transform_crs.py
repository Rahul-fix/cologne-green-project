import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import os
from tqdm import tqdm

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
WEB_DIR = DATA_DIR / "web_ready"

# Create output directory
WEB_DIR.mkdir(parents=True, exist_ok=True)

def reproject_to_web(input_path, output_path):
    """
    Reprojects a raster to EPSG:4326 (WGS84) for web visualization.
    """
    dst_crs = 'EPSG:4326'

    with rasterio.open(input_path) as src:
        # Sanitize CRS
        src_crs = src.crs
        try:
            # Force EPSG:25832 for everything that isn't already a standard EPSG code
            # This covers the EngineeringCRS case and any other weirdness
            if not src_crs.is_epsg_code:
                # print(f"Sanitizing CRS for {input_path.name}...")
                src_crs = rasterio.crs.CRS.from_epsg(25832)
        except:
            # If checking is_epsg_code fails, just force it
            src_crs = rasterio.crs.CRS.from_epsg(25832)

        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs, # Use sanitized CRS
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest if "mask" in str(input_path) else Resampling.bilinear
                )

def main():
    # Find all processed files
    files = list(PROCESSED_DIR.glob("*.tif"))
    
    print(f"Found {len(files)} files to reproject...")
    
    for file_path in tqdm(files):
        output_filename = file_path.stem + "_web.tif"
        output_path = WEB_DIR / output_filename
        
        if output_path.exists():
            continue
            
        try:
            reproject_to_web(file_path, output_path)
        except Exception as e:
            print(f"Error reprojecting {file_path.name}: {e}")

    print("Done! Web-ready files are in data/web_ready/")

if __name__ == "__main__":
    main()
