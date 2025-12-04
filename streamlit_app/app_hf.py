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
import os
import shapely.wkb
import matplotlib.colors as mcolors
from rasterio.enums import Resampling
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

# Load environment variables for local testing
load_dotenv()
load_dotenv("DL_cologne_green/.env")

st.set_page_config(page_title="GreenCologne (Cloud)", layout="wide")

st.title("🌿 GreenCologne (Cloud Dashboard)")

# --- Configuration ---
# --- Configuration ---
# Get secrets from Streamlit secrets (HF Spaces) or Environment Variables
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    DATASET_ID = st.secrets.get("DATASET_ID")
except Exception:
    HF_TOKEN = None
    DATASET_ID = None

# Fallback to environment variables (loaded via dotenv)
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not DATASET_ID:
    DATASET_ID = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN not found. If the dataset is private, you must set it in Secrets or .env.")

# Paths (HfFileSystem style)
# Use the 'hf://' protocol which maps to the registered filesystem
BASE_URL = f"hf://datasets/{DATASET_ID}"

STATS_FILE = f"{BASE_URL}/data/stats/stats.parquet"
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"
BOROUGHS_FILE = f"{BASE_URL}/data/boundaries/Stadtbezirke.parquet"
PROCESSED_PREFIX = f"{BASE_URL}/data/processed"

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    # Configure Hugging Face FileSystem
    # This avoids manual HTTP header configuration and handles auth robustly
    fs = HfFileSystem(token=HF_TOKEN)
    con.register_filesystem(fs)
        
    return con

try:
    con = get_db_connection()
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# --- Data Loading with SQL ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_district_stats():
    try:
        query = f"""
            SELECT 
                v.name, 
                ST_AsWKB(v.geometry) as geometry, 
                COALESCE(s.green_area_m2, 0) as green_area_m2,
                v.Shape_Area
            FROM '{DISTRICTS_FILE}' v 
            LEFT JOIN '{STATS_FILE}' s ON v.name = s.name
            ORDER BY green_area_m2 DESC
        """
        df = con.execute(query).fetchdf()
        
        def safe_load_wkb(x):
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None

        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        gdf['green_pct'] = (gdf['green_area_m2'] / gdf['Shape_Area']) * 100
        return gdf
    except Exception as e:
        st.error(f"Error loading stats: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_city_outline():
    try:
        query = f"""
            SELECT ST_AsWKB(ST_Union_Agg(geometry)) as geometry
            FROM '{BOROUGHS_FILE}'
        """
        df = con.execute(query).fetchdf()
        
        def safe_load_wkb(x):
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None
                
        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Buffer fix (reprojecting in memory)
        gdf = gdf.to_crs("EPSG:25832")
        gdf['geometry'] = gdf['geometry'].buffer(10).buffer(-10)
        gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
    except Exception as e:
        st.error(f"Error loading outline: {e}")
        return gpd.GeoDataFrame()

# Load Data
with st.spinner("Loading data from cloud..."):
    gdf_districts = load_district_stats()
    gdf_city = load_city_outline()

if gdf_districts.empty:
    st.warning("Could not load data. Check your bucket name and permissions.")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Statistics", "🗺️ Map Visualization"])

with tab1:
    st.subheader("Green Area by Veedel")
    
    # Metrics
    total_green_m2 = gdf_districts['green_area_m2'].sum()
    total_green_ha = total_green_m2 / 10000
    st.metric("Total Green Area", f"{total_green_ha:.2f} ha")

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

with tab2:
    st.subheader("Satellite Imagery Analysis")
    st.info("Note: Raster overlays are fetched directly from Hugging Face. Performance depends on your internet connection.")
    
    # Find processed files (masks) - We need to list them from the filesystem
    # Since we are using HfFileSystem, we can list files efficiently
    @st.cache_data(ttl=3600)
    def list_remote_files():
        fs = HfFileSystem(token=HF_TOKEN)
        # List processed files
        processed_files = fs.glob(f"datasets/{DATASET_ID}/data/processed/*_mask.tif")
        return [Path(f).stem.replace("_mask", "") for f in processed_files]

    available_tiles = list_remote_files()
    
    if not available_tiles:
        st.warning("No processed images found in the dataset.")
    else:
        available_tiles = sorted(list(set(available_tiles)))
        
        # --- Tile Selection ---
        selected_tile = st.selectbox("Select Tile", available_tiles)
        
        # Layer selection
        layer_type = st.radio("Select Layer", ["NDVI", "Segmentation Mask", "Raw Satellite (RGB)"], horizontal=True)
        
        # Helper to construct remote URL
        def get_remote_url(tile_name, file_type):
            # file_type: 'mask', 'ndvi', 'raw'
            if file_type == 'raw':
                return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/raw/{tile_name}.jp2"
            elif file_type == 'mask':
                return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/processed/{tile_name}_mask.tif"
            elif file_type == 'ndvi':
                return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data/processed/{tile_name}_ndvi.tif"
            return None

        # Helper to get bounds
        def get_remote_bounds(tile_name):
            # Try mask first, then raw
            url = get_remote_url(tile_name, 'mask')
            try:
                with rasterio.open(url) as src:
                    return src.bounds, src.crs
            except Exception:
                # Try raw
                url = get_remote_url(tile_name, 'raw')
                try:
                    with rasterio.open(url) as src:
                        return src.bounds, src.crs
                except Exception:
                    return None, None

        bounds = None
        crs = None
        if selected_tile:
            with st.spinner(f"Fetching metadata for {selected_tile}..."):
                 bounds, crs = get_remote_bounds(selected_tile)
        
        if bounds and crs:
            # Reproject bounds to WGS84 for Folium
            try:
                wgs84_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
            except Exception:
                # Fallback for "EngineeringCRS"
                try:
                    src_crs = rasterio.crs.CRS.from_epsg(25832)
                    wgs84_bounds = transform_bounds(src_crs, 'EPSG:4326', *bounds)
                except Exception as e:
                    st.error(f"Failed to reproject bounds: {e}")
                    wgs84_bounds = None

            if wgs84_bounds:
                center_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
                center_lon = (wgs84_bounds[0] + wgs84_bounds[2]) / 2
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
                
                # Base Maps
                folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
                folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Satellite (Google)", overlay=False, control=True, show=False).add_to(m)

                # Overlay Logic
                overlay_url = None
                if layer_type == "Segmentation Mask":
                    overlay_url = get_remote_url(selected_tile, 'mask')
                elif layer_type == "NDVI":
                    overlay_url = get_remote_url(selected_tile, 'ndvi')
                elif layer_type == "Raw Satellite (RGB)":
                    overlay_url = get_remote_url(selected_tile, 'raw')
                
                if overlay_url:
                    with st.spinner(f"Loading {layer_type}..."):
                        try:
                            with rasterio.open(overlay_url) as src:
                                # Downsample for performance
                                MAX_DIM = 800
                                scale = MAX_DIM / max(src.width, src.height)
                                if scale < 1:
                                    out_shape = (src.count, int(src.height * scale), int(src.width * scale))
                                    data = src.read(out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest if layer_type == "Segmentation Mask" else rasterio.enums.Resampling.bilinear)
                                else:
                                    data = src.read()
                                
                                image_data = None
                                opacity = 0.7
                                
                                if layer_type == "Segmentation Mask":
                                    mask_data = data[0]
                                    rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                                    rgba[mask_data == 1] = [0, 100, 0, 255] # Trees
                                    rgba[mask_data == 2] = [144, 238, 144, 255] # Low Veg
                                    image_data = rgba
                                    opacity = 0.9
                                    
                                elif layer_type == "NDVI":
                                    # If NDVI file exists, it's single band float
                                    # If we are computing on the fly from raw, we need raw logic
                                    # Here we assume pre-calculated NDVI exists or we read raw
                                    if src.count == 1:
                                        ndvi_data = data[0]
                                        # Handle Int16 scaling if needed
                                        if ndvi_data.dtype == 'int16':
                                            ndvi_data = ndvi_data.astype('float32') * 0.0001
                                            
                                        import matplotlib.colors as mcolors
                                        norm = mcolors.Normalize(vmin=-1, vmax=1)
                                        cmap = plt.get_cmap('RdYlGn')
                                        image_data = cmap(norm(ndvi_data))
                                    else:
                                        st.warning("NDVI file has unexpected band count.")

                                elif layer_type == "Raw Satellite (RGB)":
                                    if data.shape[0] >= 3:
                                        r = data[0]; g = data[1]; b = data[2]
                                        rgb = np.dstack((r, g, b))
                                        if src.dtypes[0] == 'uint8':
                                            image_data = rgb / 255.0
                                        else:
                                            p2, p98 = np.percentile(rgb, (2, 98))
                                            image_data = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                                        opacity = 1.0

                                if image_data is not None:
                                    folium.raster_layers.ImageOverlay(
                                        image=image_data,
                                        bounds=[[wgs84_bounds[1], wgs84_bounds[0]], [wgs84_bounds[3], wgs84_bounds[2]]],
                                        opacity=opacity,
                                        name=f"{layer_type} - {selected_tile}"
                                    ).add_to(m)

                        except Exception as e:
                            st.error(f"Error loading raster: {e}")

                folium.LayerControl().add_to(m)
                st_folium(m, width=800, height=600)
