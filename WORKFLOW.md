# Complete Database-Integrated Workflow for Cologne Green Space Analysis

This guide provides a step-by-step workflow for analyzing green spaces in Cologne using **GeoPackage as your database**. It covers data acquisition, preprocessing in QGIS, and analysis using both SQL and Python.

## **Setup: Create Your Project Structure**

Ensure your project folder is structured as follows:
```
~/cologne_green_project/
├── data/
│   ├── boundaries/          # Downloaded shapefiles
│   ├── satellite/           # Sentinel-2 bands
│   └── outputs/             # Results
├── Notebooks/               # Python analysis notebooks
└── cologne_analysis.gpkg    # Your GeoPackage database (will be created)
```

***

## **PHASE 1: Download & Import Data to GeoPackage**

### Step 1.1: Download Cologne Boundaries

1. Go to [Offene Daten Köln - Stadtviertel](https://www.offenedaten-koeln.de/dataset/stadtviertel-k%C3%B6ln)
2. Download **"Stadtviertel Shapefile ZIP"**
3. Unzip to `data/boundaries/`
   - *Note: This usually creates a folder like `Stadtviertel_19`*

### Step 1.2: Download Sentinel-2 Imagery

1. Go to [Copernicus Data Space Browser](https://dataspace.copernicus.eu/browser/)
2. Search Cologne, summer 2024, cloud < 10%, Sentinel-2 L2A
3. Download **Band 4 (Red)** and **Band 8 (NIR)** `.jp2` files
4. Save to `data/satellite/R10m/`
   - *Files will look like:*
     - `T31UGS_20250403T104041_B04_10m.jp2`
     - `T31UGS_20250403T104041_B08_10m.jp2`

### Step 1.3: Import to GeoPackage (Database)

**In QGIS:**

1. Open QGIS
2. **Set Project CRS:** Project → Properties → CRS → **EPSG:25832** (ETRS89 / UTM 32N)
3. Save project as `cologne_green_analysis.qgz`

4. **Load the Shapefile temporarily:**
   - Drag `data/boundaries/Stadtviertel_19/Stadtviertel.shp` into QGIS

5. **Export to GeoPackage (create your database):**
   - Right-click `Stadtviertel` layer → Export → **Save Features As...**
   - **Format:** GeoPackage
   - **File name:** `~/cologne_green_project/cologne_analysis.gpkg`
   - **Layer name:** `veedel_boundaries`
   - **CRS:** EPSG:25832
   - Click **OK**

6. **Remove the Shapefile layer** (you now have it in the database!)
7. **Add from GeoPackage:**
   - Layer → Add Layer → Add Vector Layer
   - Browse to `cologne_analysis.gpkg`
   - Select `veedel_boundaries` layer

### Step 1.4: Reproject Satellite Imagery to EPSG:25832

**Important:** Sentinel-2 data comes in WGS 84 / UTM zone 31N (EPSG:32631) or similar. We need to reproject it to our project CRS **EPSG:25832** (ETRS89 / UTM zone 32N) to match the Cologne boundaries.

1. **Processing → Toolbox** → Search for **"Warp (reproject)"** (GDAL)
2. **Run for Band 4 (Red):**
   - **Input layer:** `T31UGS_..._B04_10m.jp2`
   - **Target CRS:** `EPSG:25832`
   - **Resampling method:** Bilinear (better for continuous data)
   - **Output file:** `data/outputs/T31UGS_B04_reprojected.tif`
   - Click **Run**
3. **Run for Band 8 (NIR):**
   - **Input layer:** `T31UGS_..._B08_10m.jp2`
   - **Target CRS:** `EPSG:25832`
   - **Resampling method:** Bilinear
   - **Output file:** `data/outputs/T31UGS_B08_reprojected.tif`
   - Click **Run**

✅ **Now use these `_reprojected.tif` files for all analysis!**

***

## **PHASE 2: Calculate NDVI (Raster)**

### Step 2.1: Load Satellite Bands

1. Drag the **reprojected** bands into QGIS if they aren't already there:
   - `B04_reprojected.tif`
   - `B08_reprojected.tif`

### Step 2.2: Calculate NDVI

1. **Raster → Raster Calculator**
2. **Formula:**
   ```
   ("B08_reprojected@1" - "B04_reprojected@1") / ("B08_reprojected@1" + "B04_reprojected@1")
   ```
   *(Double-click the bands in the "Raster bands" list to insert exact names)*
3. **Output:** `data/outputs/ndvi_cologne.tif`
4. Click **OK**

### **Step 2.3: Style NDVI**

1. Right-click `ndvi_cologne` → Properties → Symbology
2. **Singleband pseudocolor**, color ramp **RdYlGn**, Min: -0.1, Max: 0.8

***

## **PHASE 3: Zonal Statistics → Save to Database**

### **Step 3.1: Calculate NDVI per Veedel**

**Using Processing Toolbox:**

1. **Processing → Toolbox** (search bar appears)
2. Search for **"Zonal statistics"**
3. Select **"Zonal statistics" (native QGIS algorithm)**

**Parameters:**
- **Input layer:** `veedel_boundaries` (from GeoPackage)
- **Raster layer:** `ndvi_cologne`
- **Statistics to calculate:** Check **Mean, Median, Min, Max, Std deviation**
- **Output column prefix:** `ndvi_`
- **Output:** Click the **...** button → **Save to GeoPackage**
  - **File:** `cologne_analysis.gpkg`
  - **Layer name:** `veedel_with_stats`

4. Click **Run**

✅ **Your database now contains a new table with NDVI statistics!**

***

## **PHASE 4: Analyze with SQL Queries (Database Power!)**

### **Step 4.1: Open DB Manager**

1. **Database → DB Manager → DB Manager**
2. In the tree on the left, expand **GeoPackage → cologne_analysis.gpkg**
3. You'll see your tables: `veedel_boundaries` and `veedel_with_stats`

### **Step 4.2: Run SQL Queries**

Click the **SQL Window** button (icon with database and SQL text)

**Query 1: Find the 10 Greenest Veedel**
```sql
SELECT 
    fid,
    NAME,  -- or whatever your name field is called
    ndvi_mean,
    ndvi_median
FROM veedel_with_stats
WHERE ndvi_mean IS NOT NULL
ORDER BY ndvi_mean DESC
LIMIT 10;
```

**Query 2: Find Neglected Areas (NDVI < 0.25)**
```sql
SELECT 
    fid,
    NAME,
    ndvi_mean,
    ROUND(ndvi_mean, 3) as greenness_score
FROM veedel_with_stats
WHERE ndvi_mean < 0.25
ORDER BY ndvi_mean ASC;
```

**Query 3: Create a Map Layer from Query**
```sql
SELECT 
    fid,
    geom,
    NAME,
    ndvi_mean,
    CASE 
        WHEN ndvi_mean < 0.2 THEN 'Critical'
        WHEN ndvi_mean < 0.3 THEN 'Low'
        WHEN ndvi_mean < 0.4 THEN 'Moderate'
        ELSE 'Good'
    END as green_category
FROM veedel_with_stats
WHERE ndvi_mean IS NOT NULL;
```

1. Check **☑ Load as new layer**
2. **Column with unique ID:** `fid`
3. **Geometry column:** `geom`
4. **Layer name:** `veedel_categorized`
5. Click **Load now!**

✅ **The query results are now a layer on your map!**

***

## **PHASE 5: Python Analysis (Hybrid Approach)**

We have created a dedicated notebook for this phase.

1. Open `Notebooks/cologne_analysis_workflow.ipynb`
2. Run the cells to perform the analysis using `geopandas` and `pandas`.
3. The notebook will:
    - Load data from `cologne_analysis.gpkg`
    - Calculate statistics
    - Find greenest and neglected areas
    - Save analysis results back to the GeoPackage as `veedel_analysis_final`
    - Export rankings to CSV

***

## **PHASE 6: Visualize Results**

### **Step 6.1: Create Choropleth Map**

1. Load `veedel_analysis_final` from your GeoPackage (created in Phase 5)
2. Right-click → Properties → Symbology → **Graduated**
3. **Value:** `ndvi_mean`
4. **Color ramp:** RdYlGn (Red to Green)
5. **Mode:** Natural Breaks (Jenks)
6. **Classes:** 5
7. **Classify → Apply**

### **Step 6.2: Create Categorical Map**

1. Same layer → Symbology → **Categorized**
2. **Value:** `green_category`
3. **Classify** (will show Critical, Low, Moderate, Good)
4. Manually set colors:
   - Critical: Red
   - Low: Orange
   - Moderate: Yellow
   - Good: Green
5. **Apply**
