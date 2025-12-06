import rasterio
from pathlib import Path

processed_dir = Path("data/processed")
mask_files = list(processed_dir.glob("*_mask.tif"))
ndvi_files = list(processed_dir.glob("*_ndvi.tif"))

if mask_files:
    with rasterio.open(mask_files[0]) as src:
        print(f"MASK: {mask_files[0].name}")
        print(f"  Shape: {src.shape}")
        print(f"  Count: {src.count}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Compression: {src.profile.get('compress', 'None')}")
        print(f"  BlockSize: {src.profile.get('blockxsize', 'N/A')}x{src.profile.get('blockysize', 'N/A')}")
        print(f"  Tiled: {src.profile.get('tiled', 'False')}")

if ndvi_files:
    with rasterio.open(ndvi_files[0]) as src:
        print(f"NDVI: {ndvi_files[0].name}")
        print(f"  Shape: {src.shape}")
        print(f"  Count: {src.count}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Compression: {src.profile.get('compress', 'None')}")
