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
    gdf = None
    if BOUNDARIES_FILE.exists():
        print(f"✅ Found {BOUNDARIES_FILE}")
        gdf = gpd.read_parquet(BOUNDARIES_FILE)
    else:
        print(f"⚠️  {BOUNDARIES_FILE} not found. Checking for Shapefiles...")
        # Look for any .shp file in the boundaries directory
        shp_files = list(DATA_DIR.glob("boundaries/*.shp"))
        if shp_files:
            shp_path = shp_files[0]
            print(f"✅ Found Shapefile: {shp_path}")
            gdf = gpd.read_file(shp_path)
            # Save as parquet for next time
            print(f"💾 Converting to {BOUNDARIES_FILE} for faster loading next time...")
            gdf.to_parquet(BOUNDARIES_FILE)
        else:
            print(f"❌ Error: No boundary files found in {DATA_DIR / 'boundaries'}")
            print("Please run scripts/download_boundaries.py first.")
            return
    
    # Reproject to UTM Zone 32N (EPSG:25832)
    print("🔄 Reprojecting to EPSG:25832 (UTM Zone 32N)...")
    gdf_utm = gdf.to_crs("EPSG:25832")
    
    # Get total bounds of the city
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    
    print(f"📍 Cologne Bounding Box (UTM 32N):")
    print(f"   Min: ({minx:.0f}, {miny:.0f})")
    print(f"   Max: ({maxx:.0f}, {maxy:.0f})")
    
    # Download metadata if missing or force update
    METADATA_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/dop_meta.zip"
    METADATA_ZIP = DATA_DIR / "metadata" / "dop_meta.zip"
    METADATA_FILE = DATA_DIR / "metadata" / "dop_nw.csv"
    
    print(f"⬇️  Downloading metadata from {METADATA_URL}...")
    import urllib.request
    import zipfile
    
    try:
        urllib.request.urlretrieve(METADATA_URL, METADATA_ZIP)
        with zipfile.ZipFile(METADATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR / "metadata")
        print("✅ Metadata downloaded and extracted.")
    except Exception as e:
        print(f"⚠️  Failed to download metadata: {e}")
        if not METADATA_FILE.exists():
            print("❌ Error: Metadata file not found and download failed.")
            return

    print("📄 Loading metadata from dop_nw.csv...")
    # Read CSV with semicolon delimiter, skipping the first 5 lines of metadata header
    # The actual header is on line 6 (index 5)
    df_meta = pd.read_csv(METADATA_FILE, sep=';', skiprows=5)
    
    # Ensure coordinates are numeric
    df_meta['Koordinatenursprung_East'] = pd.to_numeric(df_meta['Koordinatenursprung_East'], errors='coerce')
    df_meta['Koordinatenursprung_North'] = pd.to_numeric(df_meta['Koordinatenursprung_North'], errors='coerce')
    
    print("🔍 Filtering tiles within Cologne...")
    
    # Vectorized filtering
    mask = (
        (df_meta['Koordinatenursprung_East'] < maxx) & 
        (df_meta['Koordinatenursprung_East'] + 1000 > minx) &
        (df_meta['Koordinatenursprung_North'] < maxy) & 
        (df_meta['Koordinatenursprung_North'] + 1000 > miny)
    )
    
    cologne_tiles = df_meta[mask].copy()
    
    print(f"✅ Found {len(cologne_tiles)} tiles covering Cologne (before deduplication).")
    
    # Deduplicate: Keep only the latest year for each location
    # Sort by Kachelname (which includes year) descending, then drop duplicates by coordinates
    cologne_tiles = cologne_tiles.sort_values('Kachelname', ascending=False)
    cologne_tiles = cologne_tiles.drop_duplicates(subset=['Koordinatenursprung_East', 'Koordinatenursprung_North'], keep='first')
    
    print(f"✅ Found {len(cologne_tiles)} unique tiles covering Cologne (after deduplication).")
    
    def generate_url(row):
        kachelname = row['Kachelname']
        # Files are directly in the root folder, no subfolder needed
        return f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/{kachelname}.jp2"

    cologne_tiles['url'] = cologne_tiles.apply(generate_url, axis=1)
    cologne_tiles['filename'] = cologne_tiles['Kachelname'] + ".jp2"
    
    # Select relevant columns
    output_df = cologne_tiles[['Kachelname', 'filename', 'url', 'Koordinatenursprung_East', 'Koordinatenursprung_North']]
    
    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved tile list to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
