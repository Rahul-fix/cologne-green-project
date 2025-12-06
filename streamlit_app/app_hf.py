import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import duckdb
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import numpy as np
from pathlib import Path
import os
from huggingface_hub import HfFileSystem
import shapely.wkb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("DL_cologne_green/.env")

st.set_page_config(page_title="GreenCologne (Cloud)", layout="wide")
st.title("🌿 GreenCologne (Cloud Dashboard)")

# --- Configuration ---
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    DATASET_ID = st.secrets.get("DATASET_ID")
except Exception:
    HF_TOKEN = None
    DATASET_ID = None

if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not DATASET_ID:
    DATASET_ID = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN not found. If the dataset is private, you must set it in Secrets or .env.")

# Paths
BASE_URL = f"hf://datasets/{DATASET_ID}"
STATS_FILE = f"{BASE_URL}/data/stats/extended_stats.parquet"
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"
BOROUGHS_FILE = f"{BASE_URL}/data/boundaries/Stadtbezirke.parquet"

# --- Database ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if HF_TOKEN:
        fs = HfFileSystem(token=HF_TOKEN)
        con.register_filesystem(fs)
    return con

try:
    con = get_db_connection()
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_quarters():
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
        
        if df.empty:
            query_fallback = f"SELECT name, ST_AsWKB(geometry) as geometry, 0 as green_area_m2, Shape_Area FROM '{DISTRICTS_FILE}'"
            df = con.execute(query_fallback).fetchdf()

        def safe_load_wkb(x):
            try: return shapely.wkb.loads(bytes(x))
            except: return None

        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        if not gdf.empty and gdf.total_bounds[0] > 180:
             gdf.crs = "EPSG:25832"
             gdf = gdf.to_crs("EPSG:4326")

        return gdf
    except Exception as e:
        st.error(f"Error loading quarters: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_boroughs():
    try:
        query = f"SELECT STB_NAME, ST_AsWKB(geometry) as geometry FROM '{BOROUGHS_FILE}'"
        df = con.execute(query).fetchdf()
        
        def safe_load_wkb(x):
            try: return shapely.wkb.loads(bytes(x))
            except: return None
                
        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        if not gdf.empty and gdf.total_bounds[0] > 180:
             gdf.crs = "EPSG:25832"
             gdf = gdf.to_crs("EPSG:4326")
        
        if 'STB_NAME' in gdf.columns:
            gdf = gdf.rename(columns={'STB_NAME': 'name'})
            
        return gdf
    except Exception as e:
        st.error(f"Error loading boroughs: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_stats():
    try:
        return con.execute(f"SELECT * FROM '{STATS_FILE}'").fetchdf()
    except: return pd.DataFrame()

with st.spinner("Loading data from cloud..."):
    gdf_quarters = load_quarters()
    gdf_boroughs = load_boroughs()
    df_stats = load_stats()

# --- Colors (QML) ---
FLAIR_COLORS = {
    0: [206, 112, 121, 255],   # Building
    1: [185, 226, 212, 255],   # Greenhouse
    2: [98, 208, 255, 255],    # Swimming pool
    3: [166, 170, 183, 255],   # Impervious surface
    4: [152, 119, 82, 255],    # Pervious surface
    5: [187, 176, 150, 255],   # Bare soil
    6: [51, 117, 161, 255],    # Water
    7: [233, 239, 254, 255],   # Snow
    8: [140, 215, 106, 255],   # Herbaceous vegetation
    9: [222, 207, 85, 255],    # Agricultural land
    10: [208, 163, 73, 255],   # Plowed land
    11: [176, 130, 144, 255],  # Vineyard
    12: [76, 145, 41, 255],    # Deciduous
    13: [18, 100, 33, 255],    # Coniferous
    14: [181, 195, 53, 255],   # Brushwood
    15: [228, 142, 77, 255],   # Clear cut
    16: [34, 34, 34, 255],     # Ligneous
    17: [34, 34, 34, 255],     # Mixed
    18: [34, 34, 34, 255],     # Other
}

CLASS_LABELS = {
    0: 'Building', 1: 'Greenhouse', 2: 'Swimming pool',
    3: 'Impervious surface', 4: 'Pervious surface', 5: 'Bare soil',
    6: 'Water', 7: 'Snow', 8: 'Herbaceous vegetation',
    9: 'Agricultural land', 10: 'Plowed land', 11: 'Vineyard',
    12: 'Deciduous', 13: 'Coniferous', 14: 'Brushwood',
    15: 'Clear cut', 16: 'Ligneous', 17: 'Mixed', 18: 'Other'
}

@st.cache_data(ttl=3600)
def get_tile_to_veedel_mapping():
    try:
        tiles_url = f"hf://datasets/{DATASET_ID}/data/metadata/cologne_tiles.csv"
        tiles_df = pd.read_csv(tiles_url)
        from shapely.geometry import box
        geometries = [box(r['Koordinatenursprung_East'], r['Koordinatenursprung_North'], r['Koordinatenursprung_East']+1000, r['Koordinatenursprung_North']+1000) for _, r in tiles_df.iterrows()]
        tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
        
        q_url = f"hf://datasets/{DATASET_ID}/data/boundaries/Stadtviertel.parquet"
        q_gdf = gpd.read_parquet(q_url)
        if q_gdf.crs != "EPSG:25832": q_gdf = q_gdf.to_crs("EPSG:25832")
            
        joined = gpd.sjoin(tiles_gdf, q_gdf, how="inner", predicate="intersects")
        return joined.groupby('name')['Kachelname'].apply(list).to_dict()
    except: return {}

tile_mapping = get_tile_to_veedel_mapping()

@st.cache_data(ttl=3600)
def list_remote_files():
    fs = HfFileSystem(token=HF_TOKEN)
    processed_files = fs.glob(f"datasets/{DATASET_ID}/data/processed/*_mask.tif")
    return [Path(f).stem.replace("_mask", "") for f in processed_files]

available_tiles = list_remote_files()

if 'selected_veedel' not in st.session_state: st.session_state['selected_veedel'] = "All"
if 'map_center' not in st.session_state: st.session_state['map_center'] = [50.9375, 6.9603]
if 'map_zoom' not in st.session_state: st.session_state['map_zoom'] = 10

# --- Layout ---
col_map, col_details = st.columns([0.65, 0.35], gap="medium")

with col_details:
    st.markdown("### GreenCologne (Cloud)")
    tab_opts, tab_stats = st.tabs(["🛠️ Options", "📊 Statistics"])
    
    veedel_list = ["All"] + sorted(gdf_quarters['name'].unique().tolist()) if not gdf_quarters.empty else ["All"]
    
    with tab_opts:
        def update_zoom_for_veedel(veedel_name):
             if veedel_name == "All":
                st.session_state['map_center'] = [50.9375, 6.9603]
                st.session_state['map_zoom'] = 10
             elif not gdf_quarters.empty:
                 match = gdf_quarters[gdf_quarters['name'] == veedel_name]
                 if not match.empty:
                     centroid = match.geometry.centroid.iloc[0]
                     st.session_state['map_center'] = [centroid.y, centroid.x]
                     st.session_state['map_zoom'] = 13

        def on_veedel_change():
            sel = st.session_state['selected_veedel_widget']
            st.session_state['selected_veedel'] = sel
            update_zoom_for_veedel(sel)
        
        selected_veedel = st.selectbox(
            "Select Quarter (Veedel/Stadtviertel):", 
            veedel_list, 
            key='selected_veedel_widget', 
            on_change=on_veedel_change,
            index=veedel_list.index(st.session_state['selected_veedel']) if st.session_state['selected_veedel'] in veedel_list else 0
        )

        layer_type = st.radio(
            "Select Layer:",
            ["Segmentation Mask (Green Highlight)", "Land Cover Classes", "NDVI", "Raw Satellite (RGB)"],
            index=0
        )
        
        if layer_type == "Land Cover Classes":
            st.markdown("#### Legend")
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 12px;'>"
            for cls_id, label in CLASS_LABELS.items():
                c = FLAIR_COLORS[cls_id]
                color_css = f"rgba({c[0]},{c[1]},{c[2]},{c[3]/255})"
                legend_html += f"<div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: {color_css}; margin-right: 5px; border: 1px solid #ccc;'></div>{label}</div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
            
        current_veedel_tiles = []
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            filtered_tiles = [t for t in veedel_tiles if t in available_tiles]
            current_veedel_tiles = sorted(filtered_tiles)
        else:
            current_veedel_tiles = sorted(available_tiles)

        tile_options = ["- Select a Tile -"] + current_veedel_tiles if selected_veedel == "All" else current_veedel_tiles
        selected_tile = st.selectbox("Select Tile:", tile_options)
        if selected_tile == "- Select a Tile -": selected_tile = None
        
        st.info("ℹ️ Select a specific Veedel or Tile to view satellite imagery.")

    # --- Stats ---
    with tab_stats:
        if selected_veedel != "All" and not gdf_quarters.empty:
            row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
            if not row.empty:
                st.markdown(f"#### {selected_veedel}")
                area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row.columns else 0
                area_ha = area_m2 / 10000
                total_area_m2 = row['Shape_Area'].values[0] if 'Shape_Area' in row.columns else 1
                pct = (area_m2 / total_area_m2) * 100
                
                c1, c2 = st.columns(2)
                c1.metric("Green Area", f"{area_ha:.2f} ha")
                c2.metric("Green Coverage", f"{pct:.1f}%")
                
                fig_gauge = px.pie(
                    names=['Green', 'Other'], 
                    values=[area_m2, total_area_m2 - area_m2],
                    color=['Green', 'Other'],
                    color_discrete_map={'Green': 'green', 'Other': 'lightgray'},
                    hole=0.6, title="Green Coverage"
                )
                fig_gauge.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

        elif not df_stats.empty:
             st.markdown("#### City Overview")
             total_ha = df_stats['green_area_m2'].sum() / 10000
             st.metric("Total Green Area", f"{total_ha:.2f} ha")
             
             if not gdf_quarters.empty:
                 top_10 = gdf_quarters.sort_values('green_area_m2', ascending=False).head(10)
                 fig = px.bar(
                    top_10, x='name', y='green_area_m2',
                    title="Top 10 Greenest Quarters",
                    labels={'green_area_m2': 'Green Area (m²)', 'name': ''},
                    color='green_area_m2', color_continuous_scale='Greens'
                )
                 fig.update_layout(height=400, margin=dict(l=0, r=0))
                 st.plotly_chart(fig, use_container_width=True)

# --- Map ---
with col_map:
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    
    # 1. Districts
    if not gdf_boroughs.empty:
        folium.GeoJson(
            gdf_boroughs,
            name="Districts",
            style_function=lambda x: {'fillColor': 'none', 'color': '#333333', 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.0},
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Bezirk:'])
        ).add_to(m)
    
    # 2. Veedel
    if not gdf_quarters.empty:
        def style_fn(feature):
            name = feature['properties']['name']
            if selected_veedel != "All" and name == selected_veedel:
                 return {'fillColor': '#ffff00', 'color': 'black', 'weight': 3, 'fillOpacity': 0.0}
            green_area = feature['properties'].get('green_area_m2', 0)
            return {
                'fillColor': 'green' if green_area > 0 else 'gray',
                'color': '#666666',
                'weight': 1,
                'fillOpacity': 0.1 if selected_veedel != "All" else 0.4
            }
        
        folium.GeoJson(
            gdf_quarters,
            name="Veedel",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=['name', 'green_area_m2'], aliases=['Veedel:', 'Green Area (m²):'], localize=True)
        ).add_to(m)

    # 3. Tiles
    tiles_to_display = []
    if selected_tile: tiles_to_display = [selected_tile]
    elif selected_veedel != "All" and current_veedel_tiles:
         if len(current_veedel_tiles) > 5:
             st.toast(f"Displaying 5/{len(current_veedel_tiles)} tiles. Select specific tile for more.", icon="ℹ️")
             tiles_to_display = current_veedel_tiles[:5]
         else:
             tiles_to_display = current_veedel_tiles
    
    def open_bytes(tile_name, layer):
        fs = HfFileSystem(token=HF_TOKEN)
        path = ""
        # Priority: Web Optimized -> Processed -> Raw
        if layer == 'web_optimized': path = f"datasets/{DATASET_ID}/data/web_optimized/{tile_name}.tif"
        elif layer == 'web_optimized_mask': path = f"datasets/{DATASET_ID}/data/web_optimized/{tile_name}_mask.tif"
        elif layer == 'web_optimized_ndvi': path = f"datasets/{DATASET_ID}/data/web_optimized/{tile_name}_ndvi.tif"
        elif layer == 'raw': path = f"datasets/{DATASET_ID}/data/raw/{tile_name}.jp2"
        elif layer == 'mask': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_mask.tif"
        elif layer == 'ndvi': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_ndvi.tif"
        
        try:
            with fs.open(path, "rb") as f: return f.read()
        except: return None
    
    # FeatureGroup for dynamic layer
    fg = folium.FeatureGroup(name=layer_type, show=True)

    for tile_name in tiles_to_display:
        ftype = 'mask'
        if layer_type in ["Segmentation Mask (Green Highlight)", "Land Cover Classes"]:
             # Try optimized mask first
             ftype = 'web_optimized_mask'
        elif layer_type == 'NDVI': 
             ftype = 'web_optimized_ndvi'
        elif layer_type == 'Raw Satellite (RGB)': 
             ftype = 'web_optimized' 
        
        b = open_bytes(tile_name, ftype)
        
        # Fallbacks (Optimized -> Standard -> Raw)
        if not b:
            if ftype == 'web_optimized': b = open_bytes(tile_name, 'raw')
            elif ftype == 'web_optimized_mask': b = open_bytes(tile_name, 'mask')
            elif ftype == 'web_optimized_ndvi': 
                b = open_bytes(tile_name, 'ndvi')
                if not b: b = open_bytes(tile_name, 'raw') # NDVI needs raw if processed missing
        
        if b:
            try:
                with rasterio.MemoryFile(b) as memfile:
                    with memfile.open() as src:
                        bounds = src.bounds
                        crs = src.crs
                        # Native Bounds (EPSG:25832)
                        # We transform BOUNDS to WGS84 for Leaflet, but keep IMAGE in 25832 structure.
                        # This "stretches" it slightly but avoids pixel loss/blur from reprojection.
                        try:
                             from rasterio.crs import CRS
                             if not crs or not crs.is_epsg_code: crs = CRS.from_epsg(25832)
                             wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
                        except: continue

                        data = src.read()
                        image_data = None
                        opacity = 0.7
                        
                        if layer_type == "Segmentation Mask (Green Highlight)":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for c in [8, 9, 10, 11, 12, 13, 14]: rgba[mask == c] = [0, 255, 0, 200]
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "Land Cover Classes":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for cls_id, col in FLAIR_COLORS.items(): rgba[mask == cls_id] = col
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "NDVI":
                            if src.count == 1:
                                ndvi = data[0].astype('float32')
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-0.2, vmax=1)(ndvi))
                            else:
                                r, nir = data[0].astype('float32'), data[3].astype('float32')
                                ndvi = (nir - r) / (nir + r + 1e-8)
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-0.2, vmax=1)(ndvi))
                        elif layer_type == "Raw Satellite (RGB)":
                            if src.count >= 3:
                                rgb = np.dstack((data[0], data[1], data[2]))
                                # Handle uint16 raw
                                if rgb.dtype == 'uint16':
                                     p2, p98 = np.percentile(rgb, (2, 98))
                                     rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                                image_data = rgb
                                opacity = 1.0
                                
                        if image_data is not None:
                            folium.raster_layers.ImageOverlay(
                                image=image_data,
                                bounds=[[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]],
                                opacity=opacity,
                                name=f"{layer_type} - {tile_name}",
                                control=False
                            ).add_to(fg)
            except Exception as e:
                pass # print(e)
    fg.add_to(m)

    folium.LayerControl().add_to(m)
    
    # Click Logic
    map_output = st_folium(m, width=None, height=700, key="main_map_hf", use_container_width=True, returned_objects=["last_object_clicked"])
    
    if map_output['last_object_clicked']:
        props = map_output['last_object_clicked'].get('properties')
        if props and 'name' in props:
            clicked_name = props['name']
            if clicked_name in veedel_list and clicked_name != st.session_state['selected_veedel']:
                st.session_state['selected_veedel_widget'] = clicked_name
                st.session_state['selected_veedel'] = clicked_name
                if not gdf_quarters.empty:
                     match = gdf_quarters[gdf_quarters['name'] == clicked_name]
                     if not match.empty:
                         centroid = match.geometry.centroid.iloc[0]
                         st.session_state['map_center'] = [centroid.y, centroid.x]
                         st.session_state['map_zoom'] = 13
                st.rerun()
