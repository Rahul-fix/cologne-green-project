import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box

# Paths
DATA_DIR = Path("data")
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"
DISTRICTS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
WEB_READY_DIR = DATA_DIR / "web_ready"

def main():
    print("--- Data Availability Diagnosis ---")
    
    # 1. Check Metadata
    if not TILES_METADATA_FILE.exists():
        print("❌ Metadata file missing!")
        return
    
    tiles_df = pd.read_csv(TILES_METADATA_FILE)
    print(f"Total tiles in metadata: {len(tiles_df)}")
    
    # 2. Check Local Files
    processed_files = set(f.stem.replace("_mask", "").replace("_ndvi", "") for f in PROCESSED_DIR.glob("*.tif"))
    raw_files = set(f.stem for f in RAW_DIR.glob("*.jp2"))
    web_files = set(f.stem.replace("_mask_web", "").replace("_ndvi_web", "") for f in WEB_READY_DIR.glob("*.tif"))
    
    all_local_tiles = processed_files.union(raw_files).union(web_files)
    print(f"Total unique local tiles found: {len(all_local_tiles)}")
    print(f"  - Processed: {len(processed_files)}")
    print(f"  - Raw: {len(raw_files)}")
    print(f"  - Web Ready: {len(web_files)}")
    
    # 3. Check Mapping
    if not DISTRICTS_FILE.exists():
        print("❌ Districts file missing!")
        return

    # Recreate mapping logic
    geometries = []
    for _, row in tiles_df.iterrows():
        e = row['Koordinatenursprung_East']
        n = row['Koordinatenursprung_North']
        geometries.append(box(e, n, e + 1000, n + 1000))
    
    tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
    districts_gdf = gpd.read_parquet(DISTRICTS_FILE)
    if districts_gdf.crs != "EPSG:25832":
        districts_gdf = districts_gdf.to_crs("EPSG:25832")
        
    joined = gpd.sjoin(tiles_gdf, districts_gdf, how="inner", predicate="intersects")
    mapping = joined.groupby('name')['Kachelname'].apply(list).to_dict()
    
    print(f"\nMapped Veedels: {len(mapping)} / {len(districts_gdf)}")
    
    # 4. Analyze specific Veedels
    print("\n--- Analysis by Veedel (Sample) ---")
    
    # Sort by number of available tiles
    veedel_stats = []
    for veedel, expected_tiles in mapping.items():
        expected_set = set(expected_tiles)
        available_count = len([t for t in expected_set if t in all_local_tiles])
        veedel_stats.append({
            "veedel": veedel,
            "expected": len(expected_set),
            "available": available_count,
            "missing": len(expected_set) - available_count
        })
        
    veedel_stats_df = pd.DataFrame(veedel_stats).sort_values('available', ascending=False)
    
    print("Top 5 Veedels with most available tiles:")
    print(veedel_stats_df.head(5).to_string(index=False))
    
    print("\nTop 5 Veedels with LEAST available tiles (but > 0 expected):")
    print(veedel_stats_df[veedel_stats_df['expected'] > 0].tail(5).to_string(index=False))
    
    print("\n--- Summary ---")
    fully_covered = veedel_stats_df[veedel_stats_df['missing'] == 0]
    partially_covered = veedel_stats_df[(veedel_stats_df['available'] > 0) & (veedel_stats_df['missing'] > 0)]
    no_coverage = veedel_stats_df[veedel_stats_df['available'] == 0]
    
    print(f"Fully covered Veedels: {len(fully_covered)}")
    print(f"Partially covered Veedels: {len(partially_covered)}")
    print(f"Veedels with NO tiles available: {len(no_coverage)}")

if __name__ == "__main__":
    main()
