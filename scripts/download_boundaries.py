import requests
import zipfile
import io
import geopandas as gpd
from pathlib import Path

# URL for Cologne districts (Stadtteile/Veedel)
# Source: https://offenedaten-koeln.de/dataset/stadtteile-k%C3%B6ln
URL = "https://www.offenedaten-koeln.de/sites/default/files/Stadtteil_20.zip"

DATA_DIR = Path("data")
BOUNDARIES_DIR = DATA_DIR / "boundaries"
OUTPUT_FILE = BOUNDARIES_DIR / "Stadtviertel.parquet"

def main():
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_FILE.exists():
        print(f"✅ {OUTPUT_FILE} already exists.")
        return

    print(f"⬇️  Downloading boundaries from {URL}...")
    try:
        r = requests.get(URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Extract to temporary directory or read directly
        # We need to find the .shp file
        shp_file = None
        for name in z.namelist():
            if name.endswith(".shp"):
                shp_file = name
                break
        
        if not shp_file:
            print("❌ No .shp file found in zip.")
            return
            
        print(f"📦 Extracting {shp_file}...")
        z.extractall(BOUNDARIES_DIR)
        
        shp_path = BOUNDARIES_DIR / shp_file
        
        print("🔄 Converting to Parquet...")
        gdf = gpd.read_file(shp_path)
        
        # Ensure EPSG:4326 for consistency, or keep original?
        # The project uses EPSG:25832 for processing but 4326 for map.
        # Let's save as is, or reproject if needed. 
        # The find_cologne_tiles script reprojects to 25832 anyway.
        
        gdf.to_parquet(OUTPUT_FILE)
        print(f"✅ Saved to {OUTPUT_FILE}")
        
        # Cleanup
        for f in BOUNDARIES_DIR.glob("stadtteile*"):
            if f != OUTPUT_FILE:
                f.unlink()
                
    except Exception as e:
        print(f"❌ Failed to download/convert boundaries: {e}")

if __name__ == "__main__":
    main()
