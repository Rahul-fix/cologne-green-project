#!/usr/bin/env python3
import geopandas as gpd
from pathlib import Path

DATA_DIR = Path("data/boundaries")

def convert_to_parquet(shp_name, parquet_name):
    shp_path = DATA_DIR / shp_name
    parquet_path = DATA_DIR / parquet_name
    
    print(f"Reading {shp_path}...")
    try:
        gdf = gpd.read_file(shp_path)
        # Ensure CRS is EPSG:4326 (WGS84) as per original guide, or 25832 for calculation?
        # The original guide says: "Reproject to EPSG:4326 for storage/BigQuery compatibility"
        # But for area calculation we need a projected CRS.
        # Let's stick to what the original script likely expects or convert as needed.
        # The original `02_process_all.py` seems to handle reprojection.
        # Let's just save as is or reproject to 4326 for standard.
        
        if gdf.crs.to_epsg() != 4326:
            print(f"Reprojecting from {gdf.crs} to EPSG:4326...")
            gdf = gdf.to_crs(epsg=4326)
            
        print(f"Saving to {parquet_path}...")
        gdf.to_parquet(parquet_path)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Error converting {shp_name}: {e}")

def main():
    convert_to_parquet("Stadtviertel.shp", "Stadtviertel.parquet")
    convert_to_parquet("Stadtbezirk.shp", "Stadtbezirke.parquet")

if __name__ == "__main__":
    main()
