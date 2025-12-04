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

    # --- Veedel Mapping Logic ---
    @st.cache_data(ttl=3600)
    def get_tile_to_veedel_mapping():
        """
        Creates a mapping of Veedel -> List of Tiles based on spatial intersection.
        Fetches metadata from HF.
        """
        try:
            # Load Tiles Metadata from HF
            tiles_url = f"hf://datasets/{DATASET_ID}/data/metadata/cologne_tiles.csv"
            tiles_df = pd.read_csv(tiles_url)
            
            # Create geometries for tiles (assuming 1km x 1km)
            from shapely.geometry import box
            
            geometries = []
            for _, row in tiles_df.iterrows():
                e = row['Koordinatenursprung_East']
                n = row['Koordinatenursprung_North']
                # DOP10 tiles are 1km x 1km
                geometries.append(box(e, n, e + 1000, n + 1000))
            
            tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
            
            # Load Districts (already loaded as gdf_districts, but we need it in projected CRS for intersection)
            # We can reload or use the existing one. Existing one is WGS84.
            # Let's reload from the parquet file directly to be safe and consistent
            districts_url = f"hf://datasets/{DATASET_ID}/data/boundaries/Stadtviertel.parquet"
            districts_gdf = gpd.read_parquet(districts_url)
            
            if districts_gdf.crs != "EPSG:25832":
                districts_gdf = districts_gdf.to_crs("EPSG:25832")
                
            # Spatial Join
            joined = gpd.sjoin(tiles_gdf, districts_gdf, how="inner", predicate="intersects")
            
            # Create mapping: Veedel -> [Tile Names]
            mapping = joined.groupby('name')['Kachelname'].apply(list).to_dict()
            return mapping
        except Exception as e:
            st.warning(f"Could not load tile metadata for filtering: {e}")
            return {}

    tile_mapping = get_tile_to_veedel_mapping()

    # --- Veedel Selection UI ---
    # Get all districts
    all_veedel_names = sorted(gdf_districts['name'].tolist()) if not gdf_districts.empty else []
    veedel_options = ["All"] + all_veedel_names
    
    # Session State for Map
    if 'map_center' not in st.session_state:
        st.session_state['map_center'] = [50.9375, 6.9603]
    if 'map_zoom' not in st.session_state:
        st.session_state['map_zoom'] = 11
    if 'selected_veedel' not in st.session_state:
        st.session_state['selected_veedel'] = "All"

    def on_veedel_change():
        veedel = st.session_state['selected_veedel']
        if veedel == "All":
            st.session_state['map_center'] = [50.9375, 6.9603]
            st.session_state['map_zoom'] = 11
        else:
            # Find centroid
            selected_geom = gdf_districts[gdf_districts['name'] == veedel]
            if not selected_geom.empty:
                centroid = selected_geom.geometry.centroid.iloc[0]
                st.session_state['map_center'] = [centroid.y, centroid.x]
                st.session_state['map_zoom'] = 14

    selected_veedel = st.selectbox(
        "Select Veedel (District)", 
        veedel_options, 
        key='selected_veedel',
        on_change=on_veedel_change
    )

    available_tiles = list_remote_files()
    
    if not available_tiles:
        st.warning("No processed images found in the dataset.")
    else:
        available_tiles = sorted(list(set(available_tiles)))
        
        # Filter tiles based on Veedel
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            
            if not veedel_tiles:
                st.info(f"No tiles found for {selected_veedel} in the current dataset.")
            
            # Filter available tiles
            filtered_tiles = [t for t in veedel_tiles if t in available_tiles]
            
            if not filtered_tiles and veedel_tiles:
                 st.warning(f"Tiles exist for {selected_veedel} but are not processed yet.")
                 tile_options = []
            elif not filtered_tiles:
                 tile_options = []
            else:
                tile_options = sorted(filtered_tiles)
        else:
            tile_options = available_tiles

        # --- Tile Selection ---
        selected_tile = st.selectbox("Select Tile", tile_options)
        
        # Option to show all tiles in the current list
        show_all_tiles = st.checkbox("Show all listed tiles", value=True)
        
        tiles_to_display = []
        if show_all_tiles:
            MAX_TILES_TO_SHOW = 5 # Lower limit for cloud app to be safe
            if len(tile_options) > MAX_TILES_TO_SHOW:
                st.warning(f"Showing first {MAX_TILES_TO_SHOW} tiles out of {len(tile_options)} to avoid performance issues.")
                tiles_to_display = tile_options[:MAX_TILES_TO_SHOW]
            else:
                tiles_to_display = tile_options
        else:
            if selected_tile:
                tiles_to_display = [selected_tile]
        
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
                
                # Use session state for map center if set (from Veedel selection), otherwise tile center
                # But if we just selected a tile, maybe we want to see it?
                # Logic: If Veedel selected, use Veedel center. If All, use Tile center?
                # Let's stick to the session state which is updated by Veedel selection.
                # If "All" is selected, we might want to center on the tile.
                
                map_center = st.session_state['map_center']
                map_zoom = st.session_state['map_zoom']
                
                if selected_veedel == "All" and selected_tile:
                     # If specific tile selected in "All" mode, center on it
                     map_center = [center_lat, center_lon]
                     map_zoom = 14

                m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="CartoDB positron")
                
                # Base Maps
                folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
                folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Satellite (Google)", overlay=False, control=True, show=False).add_to(m)

                # Overlay Logic (Loop through tiles_to_display)
                for tile_name in tiles_to_display:
                    overlay_url = None
                    if layer_type == "Segmentation Mask":
                        overlay_url = get_remote_url(tile_name, 'mask')
                    elif layer_type == "NDVI":
                        overlay_url = get_remote_url(tile_name, 'ndvi')
                    elif layer_type == "Raw Satellite (RGB)":
                        overlay_url = get_remote_url(tile_name, 'raw')
                    
                    if overlay_url:
                        # Get bounds for this tile
                        t_bounds, t_crs = get_remote_bounds(tile_name)
                        if not t_bounds: continue
                        
                        try:
                            # Sanitize CRS if needed (though get_remote_bounds returns what rasterio reads)
                            # We can reuse the transform logic
                            try:
                                t_wgs84_bounds = transform_bounds(t_crs, 'EPSG:4326', *t_bounds)
                            except:
                                src_crs = rasterio.crs.CRS.from_epsg(25832)
                                t_wgs84_bounds = transform_bounds(src_crs, 'EPSG:4326', *t_bounds)
                        except:
                            continue

                        try:
                            with rasterio.open(overlay_url) as src:
                                # Downsample for performance
                                MAX_DIM = 600 # Smaller for multi-tile
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
                                    if src.count == 1:
                                        ndvi_data = data[0]
                                        if ndvi_data.dtype == 'int16':
                                            ndvi_data = ndvi_data.astype('float32') * 0.0001
                                        import matplotlib.colors as mcolors
                                        norm = mcolors.Normalize(vmin=-1, vmax=1)
                                        cmap = plt.get_cmap('RdYlGn')
                                        image_data = cmap(norm(ndvi_data))

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
                                        bounds=[[t_wgs84_bounds[1], t_wgs84_bounds[0]], [t_wgs84_bounds[3], t_wgs84_bounds[2]]],
                                        opacity=opacity,
                                        name=f"{layer_type} - {tile_name}"
                                    ).add_to(m)

                        except Exception as e:
                            print(f"Error loading raster {tile_name}: {e}")

                folium.LayerControl().add_to(m)
                
                # Handle Map Clicks
                map_output = st_folium(m, width=800, height=600, returned_objects=["last_object_clicked"])
                
                if map_output['last_object_clicked']:
                    clicked_props = map_output['last_object_clicked'].get('properties')
                    if clicked_props and 'name' in clicked_props:
                        clicked_veedel = clicked_props['name']
                        if clicked_veedel != st.session_state['selected_veedel']:
                            st.session_state['selected_veedel'] = clicked_veedel
                            st.rerun()
