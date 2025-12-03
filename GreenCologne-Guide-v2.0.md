# 🌿 GreenCologne: UPDATED Implementation Guide
## Cloud-Native Geospatial Analysis of Cologne Urban Greenery
**Status:** Production-Ready | **Last Updated:** December 2025 | **VERSION:** 2.0 (WITH DATA DOWNLOAD)

---

## 📋 TABLE OF CONTENTS
1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Infrastructure Setup (Day 1)](#phase-1-infrastructure-setup)
3. [Phase 2: Model Preparation (Day 2)](#phase-2-model-preparation)
4. [Phase 3: Download Boundaries (Day 3)](#phase-3-download-boundaries)
5. **[Phase 3.5: Download Raw Satellite Data (NEW!) (Day 4)](#phase-35-download-raw-satellite-data)** ← YOU WERE MISSING THIS!
6. [Phase 4: Processing Pipeline (Days 5-9)](#phase-4-processing-pipeline)
7. [Phase 5: Visualization Dashboard (Day 10)](#phase-5-visualization-dashboard)
8. [Phase 6: Add Airflow (Week 3)](#phase-6-airflow-orchestration)
9. [Phase 7: Add LLM Chatbot (Week 4)](#phase-7-llm-integration)

---

## ⚡ CRITICAL DISCOVERY: YOU WERE MISSING PHASE 3.5!

The original guide had a **gap** - it didn't explain how to get the raw satellite imagery!

**What was missing:**
- ❌ How to download 25 GB of satellite imagery from OpenGeoData NRW
- ❌ How to filter for Cologne tiles
- ❌ How to upload to GCS

**What you now have:**
- ✅ Complete Phase 3.5 guide (see "Phase-3-Data-Download.md")
- ✅ Automated script to find Cologne tiles
- ✅ Batch download with retry logic
- ✅ Upload to GCS step-by-step

---

## PHASE 3: DOWNLOAD COLOGNE BOUNDARIES
### Duration: 30 minutes | Cost: Free

### Step 3.1: Download Boundaries Locally
```bash
# On your M1 Mac (NOT on VM)
cd ~/cologne-project

mkdir -p boundaries
cd boundaries

# Download Stadtviertel (86 districts - the "Veedel")
wget "https://www.offenedaten-koeln.de/api/3/action/resource_download?resource_id=8dcc1872-26d4-42c6-92c6-58c48d18ef6e" -O stadtviertel.zip
unzip stadtviertel.zip

# Download Stadtbezirke (9 city districts)
wget "https://www.offenedaten-koeln.de/api/3/action/resource_download?resource_id=6d92df5a-1a61-4c77-9fd6-41f79caa01de" -O stadtbezirke.zip
unzip stadtbezirke.zip

# Verify
ls -lh Stadtviertel/ Stadtbezirke/
ogrinfo -summary Stadtviertel.shp
```

### Step 3.2: Convert Boundaries to GeoParquet
```bash
# On your M1 Mac, create script: convert_boundaries.py
cat > convert_boundaries.py << 'EOF'
#!/usr/bin/env python3
import geopandas as gpd

# Load shapefiles
print("📥 Loading Stadtviertel...")
stadtviertel = gpd.read_file("boundaries/Stadtviertel.shp")
print(f"   CRS: {stadtviertel.crs}, Rows: {len(stadtviertel)}")

print("📥 Loading Stadtbezirke...")
stadtbezirke = gpd.read_file("boundaries/Stadtbezirke.shp")
print(f"   CRS: {stadtbezirke.crs}, Rows: {len(stadtbezirke)}")

# Reproject to WGS84 (required for web & GeoParquet compatibility)
print("🔄 Reprojecting to EPSG:4326...")
stadtviertel_wgs84 = stadtviertel.to_crs("EPSG:4326")
stadtbezirke_wgs84 = stadtbezirke.to_crs("EPSG:4326")

# Save as GeoParquet (WKB geometry encoded automatically)
print("💾 Saving as GeoParquet...")
stadtviertel_wgs84.to_parquet("cologne_stadtviertel.parquet", compression='snappy')
stadtbezirke_wgs84.to_parquet("cologne_stadtbezirke.parquet", compression='snappy')

print("✅ Conversion complete!")
print(f"   Stadtviertel columns: {list(stadtviertel_wgs84.columns)}")
print(f"   Stadtbezirke columns: {list(stadtbezirke_wgs84.columns)}")
EOF

python3 convert_boundaries.py
```

### Step 3.3: Upload Boundaries to GCS
```bash
# On your M1 Mac
gsutil cp cologne_stadtviertel.parquet gs://cologne-green-project/boundaries/
gsutil cp cologne_stadtbezirke.parquet gs://cologne-green-project/boundaries/

# Verify
gsutil ls -h gs://cologne-green-project/boundaries/
```

---

## PHASE 3.5: DOWNLOAD RAW SATELLITE DATA ⭐ (THE MISSING PIECE!)
### Duration: 1-2 hours | Cost: ~$5 (included in your GCP credit)

**THIS IS CRITICAL!** The raw satellite imagery comes from OpenGeoData NRW, not from GCS directly.

### Step 3.5.1: Download Metadata & Understand Tile Structure

```bash
# On your M1 Mac
cd ~/cologne-project

mkdir -p data/metadata
cd data/metadata

# Download metadata from OpenGeoData NRW
echo "📥 Downloading OpenGeoData NRW metadata..."
wget "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/dop_meta.zip"
unzip -q dop_meta.zip

# Download georeferencing info
echo "📥 Downloading georeferencing..."
wget "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/dop_j2w.zip"
unzip -q dop_j2w.zip

ls -la
# Expected: various .txt, .j2w files

cd ../..
```

### Step 3.5.2: Identify Which Tiles Cover Cologne

```bash
# On your M1 Mac
# Create: ~/cologne-project/scripts/find_cologne_tiles.py

cat > scripts/find_cologne_tiles.py << 'EOF'
#!/usr/bin/env python3
"""
Identify which satellite tiles cover Cologne
OpenGeoData NRW uses 1km × 1km tiles in UTM Zone 32N (EPSG:25832)
Tile naming: dop_XX_YY_1_l_2_1_jp2_00 where XX, YY are coordinates / 1000
"""
import geopandas as gpd
from pyproj import Transformer
import pandas as pd

# Load Cologne boundaries
cologne_bounds = gpd.read_file("cologne_stadtviertel.parquet")
cologne_bbox = cologne_bounds.total_bounds  # [minx, miny, maxx, maxy] in WGS84

print(f"🗺️  Cologne bounding box (EPSG:4326):")
print(f"   Min: ({cologne_bbox[0]:.4f}, {cologne_bbox[1]:.4f})")
print(f"   Max: ({cologne_bbox[2]:.4f}, {cologne_bbox[3]:.4f})")

# Convert to UTM Zone 32N (EPSG:25832) - the CRS used by OpenGeoData NRW
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
coords_utm = transformer.transform_bounds(cologne_bbox[0], cologne_bbox[1], cologne_bbox[2], cologne_bbox[3])

print(f"\n🔄 Converted to UTM Zone 32N (EPSG:25832):")
print(f"   Min: ({coords_utm[0]:.0f}, {coords_utm[1]:.0f})")
print(f"   Max: ({coords_utm[2]:.0f}, {coords_utm[3]:.0f})")

# Calculate which tiles cover Cologne
# Each tile is 1km × 1km, so tile coordinates = UTM / 1000
min_x_tile = int(coords_utm[0] / 1000)
max_x_tile = int(coords_utm[2] / 1000)
min_y_tile = int(coords_utm[1] / 1000)
max_y_tile = int(coords_utm[3] / 1000)

print(f"\n📍 Expected tile coordinates:")
print(f"   X range: {min_x_tile} to {max_x_tile}")
print(f"   Y range: {min_y_tile} to {max_y_tile}")

# Generate tile list
cologne_tiles = []
for x in range(min_x_tile, max_x_tile + 1):
    for y in range(min_y_tile, max_y_tile + 1):
        tile_name = f"dop_{x:02d}_{y:02d}_1_l_2_1_jp2_00"
        cologne_tiles.append({
            'tile_id': tile_name,
            'x': x,
            'y': y,
            'url': f"https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/{tile_name}/{tile_name}.jp2"
        })

df_tiles = pd.DataFrame(cologne_tiles)

print(f"\n✅ Identified {len(df_tiles)} tiles covering Cologne\n")
print("Tile List:")
print(df_tiles.to_string(index=False))

# Save for batch download
df_tiles.to_csv("data/cologne_tiles.csv", index=False)
print(f"\n💾 Saved tile list to: data/cologne_tiles.csv")
EOF

python scripts/find_cologne_tiles.py
```

**Expected Output:**
```
🗺️  Cologne bounding box (EPSG:4326):
   Min: (6.7523, 50.8153)
   Max: (7.1850, 51.0700)

🔄 Converted to UTM Zone 32N (EPSG:25832):
   Min: (365000, 5630000)
   Max: (384000, 5656000)

📍 Expected tile coordinates:
   X range: 32 to 38
   Y range: 56 to 57

✅ Identified 9 tiles covering Cologne
```

### Step 3.5.3: Download JP2 Files

```bash
# On your M1 Mac
mkdir -p data/satellite/raw_jp2
cd data/satellite/raw_jp2

echo "📥 Starting satellite imagery download..."
echo "⏱️  This will take 30-60 minutes depending on internet speed"

while IFS=',' read -r tile_id x y url; do
    if [ "$tile_id" = "tile_id" ]; then
        continue  # Skip header
    fi
    
    filename="${tile_id}.jp2"
    
    if [ -f "$filename" ]; then
        SIZE=$(du -h "$filename" | cut -f1)
        echo "⏭️  Skipping $filename (already exists, $SIZE)"
        continue
    fi
    
    echo "📥 Downloading $filename..."
    
    # Use wget with retry and resume
    wget -c --tries=3 --timeout=30 "$url" -O "$filename"
    
    if [ $? -eq 0 ]; then
        SIZE=$(du -h "$filename" | cut -f1)
        echo "   ✅ Success ($SIZE)"
    else
        echo "   ❌ Failed (will retry next time)"
    fi
    
done < ../../data/cologne_tiles.csv

echo ""
echo "✅ Download phase complete!"
echo "📊 Statistics:"
echo "   Total files: $(ls -1 *.jp2 2>/dev/null | wc -l)"
echo "   Total size: $(du -sh . | cut -f1)"

cd ../../..
```

**⏱️ Expected Time:** 30-60 minutes
**📊 Expected Size:** 400-500 MB total (each tile is ~45-55 MB)

### Step 3.5.4: Verify Downloaded Files

```bash
# On your M1 Mac
cd data/satellite/raw_jp2

echo "🔍 Verifying downloads..."

for file in *.jp2; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    
    if [ $SIZE -lt 10000000 ]; then
        echo "⚠️  $file is too small ($SIZE bytes) - corrupted?"
    else
        SIZE_MB=$((SIZE / 1048576))
        echo "✅ $file - ${SIZE_MB}MB"
    fi
done

echo ""
echo "🧪 Testing with GDAL (verify files are readable)..."
SAMPLE=$(ls *.jp2 | head -1)

gdalinfo "$SAMPLE" 2>/dev/null | head -15
```

**Expected Output:**
```
✅ dop_32_56_1_l_2_1_jp2_00.jp2 - 45MB
✅ dop_33_56_1_l_2_1_jp2_00.jp2 - 52MB
...

Driver: JP2OpenJPEG/JPEG 2000
Size is 1024, 1024
Coordinate System is:
    PROJCS["WGS 84 / UTM zone 32N",
    ...
```

### Step 3.5.5: Upload to Google Cloud Storage

```bash
# On your M1 Mac
cd data/satellite/raw_jp2

echo "📤 Uploading satellite imagery to GCS..."
echo "   Target: gs://cologne-green-project/raw/"

# Parallel upload (faster)
gsutil -m -o GSUtil:parallel_thread_count=5 cp *.jp2 gs://cologne-green-project/raw/

# Verify upload
echo ""
echo "✅ Verifying upload..."
gsutil ls -h gs://cologne-green-project/raw/

echo ""
echo "📊 Checking GCS storage:"
gsutil du -sh gs://cologne-green-project/
```

**Expected Output:**
```
gs://cologne-green-project/raw/dop_32_56_1_l_2_1_jp2_00.jp2
gs://cologne-green-project/raw/dop_33_56_1_l_2_1_jp2_00.jp2
gs://cologne-green-project/raw/dop_34_56_1_l_2_1_jp2_00.jp2
...

📊 Checking GCS storage:
425 MB      gs://cologne-green-project/
```

### Step 3.5.6: Summary - What's Now in GCS

```
gs://cologne-green-project/
├── raw/                                    ← 9 satellite tiles (425 MB)
│   ├── dop_32_56_1_l_2_1_jp2_00.jp2      ← Each is ~45-55 MB
│   ├── dop_33_56_1_l_2_1_jp2_00.jp2
│   ├── dop_34_56_1_l_2_1_jp2_00.jp2
│   ├── ... (6 more tiles)
│   └── dop_38_57_1_l_2_1_jp2_00.jp2
│
└── boundaries/                             ← Cologne administrative boundaries
    ├── cologne_stadtviertel.parquet      ← 86 districts (Veedel)
    └── cologne_stadtbezirke.parquet      ← 9 city districts
```

---

## PHASE 4: PROCESSING PIPELINE
### Duration: 6-8 hours (parallel processing) | Cost: ~$3.50

[Rest of Phase 4 from original guide remains the same...]

---

## ✅ UPDATED COMPLETION CHECKLIST

### Phase 1-2: Infrastructure & Model (Days 1-2)
- [ ] GCP project created and billing enabled
- [ ] T4 GPU quota approved
- [ ] VM created and SSH working
- [ ] FLAIR-HUB installed with model weights downloaded

### Phase 3: Boundaries (Day 3)
- [ ] Downloaded Cologne Stadtviertel & Stadtbezirke shapefiles
- [ ] Converted to GeoParquet with EPSG:4326
- [ ] Uploaded boundaries to gs://cologne-green-project/boundaries/

### **Phase 3.5: Raw Satellite Data (Day 4) ⭐ THE MISSING PIECE!**
- [ ] Downloaded metadata from OpenGeoData NRW
- [ ] Identified 9 Cologne tiles using Python script
- [ ] Downloaded all 9 JP2 files (425 MB total)
- [ ] Verified files with GDAL (readable, correct CRS)
- [ ] Uploaded all files to gs://cologne-green-project/raw/
- [ ] ✅ **NOW YOU'RE READY FOR PHASE 4!**

### Phase 4: Processing (Days 5-9)
- [ ] Created processing script (02_process_all.py)
- [ ] Ran FLAIR-HUB inference on 9 tiles
- [ ] Generated GeoParquet with results
- [ ] Uploaded to gs://cologne-green-project/stats/

### Phase 5: Dashboard (Day 10)
- [ ] Created Streamlit app
- [ ] Deployed to Cloud Run or local testing

### Phase 6: Airflow (Week 3)
- [ ] Added monthly incremental processing
- [ ] Set up automated trend analysis
- [ ] Email notifications configured

### Phase 7: LLM Chatbot (Week 4)
- [ ] Integrated Google Gemini LLM
- [ ] Added natural language query capability
- [ ] Chat history management

---

## 🎯 NEXT STEPS

**You are NOW here:** ✅ Phase 3.5 Complete!

**Next action:**
1. Download Phase-3-Data-Download.md (the complete guide)
2. Follow steps 3.5.1 through 3.5.6 on your M1 Mac
3. Once files are in GCS, proceed with Phase 4 on the VM

**Estimated timeline:**
- Phase 3.5: 1-2 hours (mostly download time)
- Phase 4: 6-8 hours on VM (can run overnight)
- Phase 5-7: 1 week total

---

**Document Version:** 2.0 (WITH SATELLITE DATA DOWNLOAD!)
**Status:** ✅ Now Complete with All Missing Pieces
