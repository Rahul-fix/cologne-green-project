#!/usr/bin/env python3
"""
Identify which satellite tiles cover Cologne.
OpenGeoData NRW uses 1km x 1km tiles in UTM Zone 32N (EPSG:25832).
Tile naming: dop_XX_YY_1_l_2_1_jp2_00 where XX, YY are coordinates / 1000.
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
OUTPUT_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

def main():
    if not BOUNDARIES_FILE.exists():
        print(f"❌ Error: {BOUNDARIES_FILE} not found.")
        return

    print("📥 Loading Cologne boundaries...")
    gdf = gpd.read_parquet(BOUNDARIES_FILE)
    
    # Reproject to UTM Zone 32N (EPSG:25832)
    print("🔄 Reprojecting to EPSG:25832 (UTM Zone 32N)...")
    gdf_utm = gdf.to_crs("EPSG:25832")
    
    # Get total bounds of the city
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    
    print(f"📍 Cologne Bounding Box (UTM 32N):")
    print(f"   Min: ({minx:.0f}, {miny:.0f})")
    print(f"   Max: ({maxx:.0f}, {maxy:.0f})")
    
    # Load metadata CSV
    METADATA_FILE = DATA_DIR / "metadata" / "dop_nw.csv"
    if not METADATA_FILE.exists():
        print(f"❌ Error: {METADATA_FILE} not found.")
        return

    print("📄 Loading metadata from dop_nw.csv...")
    # Read CSV with semicolon delimiter, skipping the first 5 lines of metadata header
    # The actual header is on line 6 (index 5)
    df_meta = pd.read_csv(METADATA_FILE, sep=';', skiprows=5)
    
    # Ensure coordinates are numeric
    df_meta['Koordinatenursprung_East'] = pd.to_numeric(df_meta['Koordinatenursprung_East'], errors='coerce')
    df_meta['Koordinatenursprung_North'] = pd.to_numeric(df_meta['Koordinatenursprung_North'], errors='coerce')
    
    # Filter tiles within the bounding box
    # We add a small buffer (e.g. 0) or just strict inequality
    # The coordinates in CSV are the top-left or bottom-left? Usually bottom-left for UTM grid.
    # Let's assume they represent the corner. A tile is 1km x 1km (1000m).
    # So a tile at (E, N) covers (E, N) to (E+1000, N+1000).
    # We want tiles that intersect the bbox.
    
    # Tile MinX < BBox MaxX AND Tile MaxX > BBox MinX
    # Tile MinY < BBox MaxY AND Tile MaxY > BBox MinY
    
    print("🔍 Filtering tiles within Cologne...")
    
    # Vectorized filtering
    mask = (
        (df_meta['Koordinatenursprung_East'] < maxx) & 
        (df_meta['Koordinatenursprung_East'] + 1000 > minx) &
        (df_meta['Koordinatenursprung_North'] < maxy) & 
        (df_meta['Koordinatenursprung_North'] + 1000 > miny)
    )
    
    cologne_tiles = df_meta[mask].copy()
    
    print(f"✅ Found {len(cologne_tiles)} tiles covering Cologne.")
    
    # Construct download URLs
    # URL format: https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/{folder_name}/{tile_name}.jp2
    # Folder name seems to be derived from tile name or coordinates.
    # From sample: dop10rgbi_32_356_5645_1_nw_2025 -> folder dop_356_5645_1_nw_2025
    # Let's look at the CSV 'Kachelname' again.
    # Example: dop10rgbi_32_478_5740_1_nw_2024
    # The folder usually matches the file name without the 'dop10rgbi_32_' prefix? 
    # Wait, the sample URL was: .../dop_356_5645_1_nw_2025/dop10rgbi_32_356_5645_1_nw_2025.jp2
    # The folder is `dop_` + `356_5645_1_nw_2025`.
    # The file is `dop10rgbi_32_` + `356_5645_1_nw_2025`.
    # So we can extract the suffix.
    
    def generate_url(row):
        kachelname = row['Kachelname']
        # Extract suffix: remove 'dop10rgbi_32_'
        if kachelname.startswith('dop10rgbi_32_'):
            suffix = kachelname.replace('dop10rgbi_32_', '')
            folder = f"dop_{suffix}"
            return f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/{folder}/{kachelname}.jp2"
        return None

    cologne_tiles['url'] = cologne_tiles.apply(generate_url, axis=1)
    cologne_tiles['filename'] = cologne_tiles['Kachelname'] + ".jp2"
    
    # Select relevant columns
    output_df = cologne_tiles[['Kachelname', 'filename', 'url', 'Koordinatenursprung_East', 'Koordinatenursprung_North']]
    
    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved tile list to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
