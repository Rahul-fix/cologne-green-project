import streamlit as st
import plotly.express as px
from pathlib import Path
import duckdb
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd

st.set_page_config(page_title="GreenCologne (DuckDB)", layout="wide")

st.title("🌿 GreenCologne (DuckDB Backend)")

# Paths
DATA_DIR = Path("data")
STATS_FILE = str(DATA_DIR / "stats" / "stats.parquet")
PROCESSED_DIR = DATA_DIR / "processed"
DISTRICTS_FILE = str(DATA_DIR / "boundaries" / "Stadtviertel.parquet")
BOROUGHS_FILE = str(DATA_DIR / "boundaries" / "Stadtbezirke.parquet")

if not Path(STATS_FILE).exists():
    st.error(f"Stats file not found at {STATS_FILE}. Run scripts/05_generate_stats.py first.")
    st.stop()

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    return con

con = get_db_connection()

# --- Data Loading with SQL ---
@st.cache_data
def load_district_stats():
    # Join stats with district boundaries directly in SQL
    # We select geometry as WKB (Well-Known Binary) to load into GeoPandas easily if needed,
    # or just keep it as is. GeoPandas reads WKB.
    query = f"""
        SELECT 
            v.name, 
            ST_AsWKB(v.geometry) as geometry, 
            COALESCE(s.green_area_m2, 0) as green_area_m2,
            v.Shape_Area
        FROM '{DISTRICTS_FILE}' v 
        LEFT JOIN '{STATS_FILE}' s ON v.name = s.name
    """
    # to show top 10 districts
    # query = f"""
    #     SELECT 
    #         v.name, 
    #         ST_AsWKB(v.geometry) as geometry, 
    #         COALESCE(s.green_area_m2, 0) as green_area_m2,
    #         v.Shape_Area
    #     FROM '{DISTRICTS_FILE}' v 
    #     LEFT JOIN '{STATS_FILE}' s ON v.name = s.name
    #     ORDER BY green_area_m2 DESC
    # """

    # DuckDB to GeoDataFrame
    # We can fetch as arrow or pandas. 
    # If we fetch as pandas, 'geometry' might be bytes (WKB).
    df = con.execute(query).fetchdf()
    
    # Convert WKB bytes to geometry objects for GeoPandas
    # Note: DuckDB spatial returns geometry as WKB blobs in Python usually
    import shapely.wkb
    
    def safe_load_wkb(x):
        try:
            return shapely.wkb.loads(bytes(x))
        except Exception:
            return None

    df['geometry'] = df['geometry'].apply(safe_load_wkb)
    # Drop invalid geometries
    df = df.dropna(subset=['geometry'])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Calculate percentage
    gdf['green_pct'] = (gdf['green_area_m2'] / gdf['Shape_Area']) * 100
    return gdf

@st.cache_data
def load_city_outline():
    # Use ST_Union_Agg to dissolve boundaries in SQL
    query = f"""
        SELECT ST_AsWKB(ST_Union_Agg(geometry)) as geometry
        FROM '{BOROUGHS_FILE}'
    """
    df = con.execute(query).fetchdf()
    
    import shapely.wkb
    def safe_load_wkb(x):
        try:
            return shapely.wkb.loads(bytes(x))
        except Exception:
            return None
            
    df['geometry'] = df['geometry'].apply(safe_load_wkb)
    df = df.dropna(subset=['geometry'])
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Fix topology artifacts (buffering) - doing this in Python for now as ST_Buffer 
    # in DuckDB might require projection handling similar to Python
    gdf = gdf.to_crs("EPSG:25832")
    gdf['geometry'] = gdf['geometry'].buffer(10).buffer(-10)
    gdf = gdf.to_crs("EPSG:4326")
    
    return gdf

# Load Data
gdf_districts = load_district_stats()
gdf_city = load_city_outline()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Statistics", "🗺️ Map Visualization"])

with tab1:
    st.subheader("Green Area by Veedel (DuckDB SQL)")

    if gdf_districts.empty:
        st.warning("No data available.")
    else:
        # Metrics
        total_green_m2 = gdf_districts['green_area_m2'].sum()
        total_green_ha = total_green_m2 / 10000
        st.metric("Total Green Area (Sample)", f"{total_green_ha:.2f} ha")

        # Chart
        top_10 = gdf_districts.head(10)
        fig = px.bar(
            top_10, 
            x='name', 
            y='green_area_m2',
            title="Top 10 Greenest Districts (m²)",
            labels={'green_area_m2': 'Green Area (m²)', 'name': 'District'},
            color='green_area_m2',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig)

        with st.expander("View Raw Data"):
            st.dataframe(gdf_districts.drop(columns='geometry'))

with tab2:
    st.subheader("Satellite Imagery Analysis")
    
    # Find processed files
    ndvi_files = list(PROCESSED_DIR.glob("*_ndvi.tif"))
    
    if not ndvi_files:
        st.warning("No processed images found in data/processed/")
    else:
        # Selector for tile
        tile_options = [f.stem.replace("_ndvi", "") for f in ndvi_files]
        selected_tile = st.selectbox("Select Tile", tile_options)
        
        # Layer selection
        layer_type = st.radio("Select Layer", ["NDVI", "Segmentation Mask"], horizontal=True)
        
        # Paths
        ndvi_path = PROCESSED_DIR / f"{selected_tile}_ndvi.tif"
        mask_path = PROCESSED_DIR / f"{selected_tile}_mask.tif"
        
        # Get bounds from one of the files to center the map
        bounds = None
        crs = None
        if ndvi_path.exists():
            with rasterio.open(ndvi_path) as src:
                bounds = src.bounds
                crs = src.crs
        
        if bounds and crs:
            # Reproject bounds to WGS84 for Folium
            try:
                wgs84_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
                center_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
                center_lon = (wgs84_bounds[0] + wgs84_bounds[2]) / 2
                
                # Create Map
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
                
                # Base Maps
                folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
                folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Satellite (Google)", overlay=False, control=True, show=False).add_to(m)

                # --- District Analysis Layer (from DuckDB) ---
                folium.Choropleth(
                    geo_data=gdf_districts,
                    name="District Analysis (Green %)",
                    data=gdf_districts,
                    columns=['name', 'green_pct'],
                    key_on='feature.properties.name',
                    fill_color='RdYlGn',
                    fill_opacity=0.6,
                    line_opacity=0.2,
                    legend_name='Green Space Percentage (%)',
                    highlight=True
                ).add_to(m)
                
                folium.GeoJson(
                    gdf_districts,
                    name="District Tooltips",
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'green_pct', 'green_area_m2'],
                        aliases=['District:', 'Green %:', 'Green Area (m²):'],
                        localize=True,
                        sticky=False
                    )
                ).add_to(m)

                # --- City Outline Layer (from DuckDB) ---
                folium.GeoJson(
                    gdf_city,
                    name="Cologne City Outline",
                    style_function=lambda x: {
                        'fillColor': '#3388ff',
                        'fillOpacity': 0.1,
                        'color': '#222222',
                        'weight': 2,
                        'opacity': 0.8
                    }
                ).add_to(m)

                # --- Image Overlays ---
                MAX_DIM = 1000
                if layer_type == "NDVI" and ndvi_path.exists():
                    with rasterio.open(ndvi_path) as src:
                        scale = min(MAX_DIM / src.width, MAX_DIM / src.height, 1.0)
                        new_shape = (int(src.height * scale), int(src.width * scale))
                        img = src.read(1, out_shape=new_shape, resampling=rasterio.enums.Resampling.bilinear)
                        
                        # Handle Int16 scaling
                        if img.dtype == np.int16:
                            img = img.astype(np.float32) * 0.0001
                        img_min, img_max = np.nanmin(img), np.nanmax(img)
                        norm_img = (img - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img)
                        cmap = plt.get_cmap("RdYlGn")
                        colored_img = cmap(norm_img)
                        img_bytes = (colored_img * 255).astype(np.uint8)
                        folium.raster_layers.ImageOverlay(
                            image=img_bytes,
                            bounds=[[wgs84_bounds[1], wgs84_bounds[0]], [wgs84_bounds[3], wgs84_bounds[2]]],
                            opacity=0.8,
                            name="NDVI Overlay"
                        ).add_to(m)
                        
                elif layer_type == "Segmentation Mask" and mask_path.exists():
                    with rasterio.open(mask_path) as src:
                        scale = min(MAX_DIM / src.width, MAX_DIM / src.height, 1.0)
                        new_shape = (int(src.height * scale), int(src.width * scale))
                        img = src.read(1, out_shape=new_shape, resampling=rasterio.enums.Resampling.nearest)
                        h, w = img.shape
                        rgba = np.zeros((h, w, 4), dtype=np.uint8)
                        mask_indices = (img == 1)
                        rgba[mask_indices, 0] = 0; rgba[mask_indices, 1] = 255; rgba[mask_indices, 2] = 0; rgba[mask_indices, 3] = 200
                        folium.raster_layers.ImageOverlay(
                            image=rgba,
                            bounds=[[wgs84_bounds[1], wgs84_bounds[0]], [wgs84_bounds[3], wgs84_bounds[2]]],
                            opacity=0.9,
                            name="Vegetation Mask"
                        ).add_to(m)

                folium.LayerControl().add_to(m)
                st_folium(m, width=800, height=600)
            
            except Exception as e:
                st.error(f"Error rendering map: {e}")
            
        else:
            st.error("Could not determine bounds for map.")

st.sidebar.info("DuckDB Spatial Backend")
