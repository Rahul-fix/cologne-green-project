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

# Load environment variables for local testing
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
# Paths
BASE_URL = f"hf://datasets/{DATASET_ID}"
STATS_FILE = f"{BASE_URL}/data/stats/extended_stats.parquet"
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"
# ... [Other Paths Unchanged] ...

# ... [Other Code Unchanged] ...

    # --- Stats Tab ---
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
                
                # Class Breakdown (New)
                class_cols = [c for c in row.index if str(c).startswith('area_')]
                if class_cols:
                    class_data = row[class_cols].T.reset_index()
                    class_data.columns = ['class_col', 'area_m2']
                    class_data['class_id'] = class_data['class_col'].str.replace('area_', '').astype(int)
                    class_data['class_name'] = class_data['class_id'].map(CLASS_LABELS)
                    class_data['color'] = class_data['class_id'].map(lambda x: f"rgba({FLAIR_COLORS[x][0]},{FLAIR_COLORS[x][1]},{FLAIR_COLORS[x][2]}, 1)")
                    
                    fig_bar = px.bar(
                        class_data, x='class_name', y='area_m2', 
                        title="Land Cover Distribution", 
                        labels={'area_m2': 'Area (m²)', 'class_name': 'Class'},
                        color='class_name', color_discrete_map={row['class_name']: row['color'] for _, row in class_data.iterrows()}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        elif not df_stats.empty:
             st.markdown("#### City Overview")
             if 'green_area_m2' in df_stats.columns:
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

# --- Map Logic ---
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
    
    # 3. Quarters (Veedel)
    if not gdf_quarters.empty:
        # Calculate bounds for NDVI coloring
        min_ndvi = gdf_quarters['ndvi_mean'].min() if 'ndvi_mean' in gdf_quarters else 0
        max_ndvi = gdf_quarters['ndvi_mean'].max() if 'ndvi_mean' in gdf_quarters else 0.6
        if pd.isna(min_ndvi): min_ndvi = 0
        if pd.isna(max_ndvi): max_ndvi = 0.6
        
        def get_color(feature):
            if selected_veedel != "All" and feature['properties']['name'] == selected_veedel:
                return '#ffff00' # Yellow highlight for selection
            
            val = feature['properties'].get('ndvi_mean')
            if val is None or pd.isna(val): return 'gray'
            
            # Normalize 0 to 1 based on min/max
            norm = (val - min_ndvi) / (max_ndvi - min_ndvi + 1e-9)
            norm = max(0, min(1, norm))
            
            # Use RdYlGn colormap (Red -> Yellow -> Green)
            # 0.0 -> Red, 1.0 -> Green
            rgba = plt.get_cmap('RdYlGn')(norm)
            return mcolors.to_hex(rgba)

        def style_fn(feature):
            name = feature['properties']['name']
            is_selected = (selected_veedel != "All" and name == selected_veedel)
            
            return {
                'fillColor': get_color(feature),
                'color': 'black' if is_selected else '#666666',
                'weight': 3 if is_selected else 1,
                'fillOpacity': 0.0 if is_selected else 0.6  # Transparent if selected, else 0.6
            }
        
        # Prepare data for tooltip (PCT calculation)
        if 'green_area_m2' in gdf_quarters.columns and 'Shape_Area' in gdf_quarters.columns:
             gdf_quarters['green_pct'] = (gdf_quarters['green_area_m2'] / gdf_quarters['Shape_Area']) * 100
        
        folium.GeoJson(
            gdf_quarters,
            name="Veedel (NDVI)",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'green_area_m2', 'green_pct', 'ndvi_mean'], 
                aliases=['Veedel:', 'Green Area (m²):', 'Green Coverage (%):', 'Mean NDVI:'], 
                localize=True,
                fmt='.2f'
            )
        ).add_to(m)

    # 4. Tiles Overlay (Single Selection)
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
        if layer == 'raw': path = f"datasets/{DATASET_ID}/data/raw/{tile_name}.jp2"
        elif layer == 'mask': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_mask.tif"
        elif layer == 'ndvi': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_ndvi.tif"
        try:
            with fs.open(path, "rb") as f: return f.read()
        except: return None
    
    # Only add the SELECTED layer
    layer_type = layer_selection
    
    # FeatureGroup for dynamic layer
    fg = folium.FeatureGroup(name=layer_type, show=True)
    
    for tile_name in tiles_to_display:
        ftype = 'mask' # default
        if layer_type == 'NDVI': ftype = 'ndvi'
        elif layer_type == 'Raw Satellite (RGB)': ftype = 'raw'
        
        b = open_bytes(tile_name, ftype)
        if not b and ftype == 'ndvi': b = open_bytes(tile_name, 'raw') 
        
        if b:
            try:
                with rasterio.MemoryFile(b) as memfile:
                    with memfile.open() as src:
                        bounds = src.bounds
                        crs = src.crs
                        try:
                                from rasterio.crs import CRS
                                if not crs.is_epsg_code: crs = CRS.from_epsg(25832)
                                wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
                        except: continue

                        data = src.read()
                        image_data = None
                        opacity = 0.7
                        
                        if layer_type == "Segmentation Mask (Green Highlight)":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for c in [4, 5, 6, 7, 9, 10, 11, 12]: rgba[mask == c] = [0, 255, 0, 200]
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "Land Cover Classes":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for cls, col in FLAIR_COLORS.items(): rgba[mask == cls] = col
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "NDVI":
                            if src.count == 1:
                                ndvi = data[0].astype('float32') * 0.0001
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-1, vmax=1)(ndvi))
                            else:
                                r, nir = data[0].astype('float32'), data[3].astype('float32')
                                ndvi = (nir - r) / (nir + r + 1e-8)
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-1, vmax=1)(ndvi))
                        elif layer_type == "Raw Satellite (RGB)":
                            if src.count >= 3:
                                rgb = np.dstack((data[0], data[1], data[2]))
                                p2, p98 = np.percentile(rgb, (2, 98))
                                image_data = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                                opacity = 1.0
                                
                        if image_data is not None:
                                folium.raster_layers.ImageOverlay(
                                image=image_data,
                                bounds=[[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]],
                                opacity=opacity,
                                name=f"{layer_type} - {tile_name}",
                                control=False
                                ).add_to(fg)
            except: pass
    fg.add_to(m)

    folium.LayerControl().add_to(m)
    
    # 5. Map Click Logic (Zoom Fix)
    map_output = st_folium(m, width=None, height=700, key="main_map_hf", use_container_width=True, returned_objects=["last_object_clicked"])
    
    if map_output['last_object_clicked']:
        props = map_output['last_object_clicked'].get('properties')
        if props and 'name' in props:
            clicked_name = props['name']
            if clicked_name in veedel_list and clicked_name != st.session_state['selected_veedel']:
                st.session_state['selected_veedel_widget'] = clicked_name
                st.session_state['selected_veedel'] = clicked_name
                # MANUAL UPDATE for next rerun with Centroid Fix
                if not gdf_quarters.empty:
                     match = gdf_quarters[gdf_quarters['name'] == clicked_name]
                     if not match.empty:
                         # Reproject to metric (EPSG:25832) for accurate centroid, then back to WGS84
                         try:
                             centroid = match.to_crs("EPSG:25832").geometry.centroid.to_crs("EPSG:4326").iloc[0]
                             st.session_state['map_center'] = [centroid.y, centroid.x]
                             st.session_state['map_zoom'] = 13
                         except Exception as e:
                             st.warning(f"Could not calc centroid: {e}")
                st.rerun()
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"
BOROUGHS_FILE = f"{BASE_URL}/data/boundaries/Stadtbezirke.parquet"

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
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
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None

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
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None
                
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

# FLAIR-HUB Color Palette & Labels
FLAIR_COLORS = {
    1: [219, 14, 154, 255],     2: [255, 0, 0, 255],        3: [143, 85, 41, 255],
    4: [0, 255, 0, 255],        5: [32, 105, 10, 255],      6: [90, 31, 10, 255],
    7: [48, 25, 23, 255],       8: [13, 227, 237, 255],     9: [2, 161, 9, 255],
    10: [136, 68, 145, 255],    11: [55, 23, 40, 255],      12: [173, 201, 198, 255],
    13: [0, 0, 0, 0],
}
CLASS_LABELS = {
    1: 'Building', 2: 'Impervious', 3: 'Barren', 4: 'Grass', 5: 'Brush',
    6: 'Agriculture', 7: 'Tree', 8: 'Water', 9: 'Herbaceous', 10: 'Shrub',
    11: 'Moss', 12: 'Lichen', 13: 'Unknown'
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
    
    # Tabs
    tab_opts, tab_stats = st.tabs(["🛠️ Options", "📊 Statistics"])
    
    veedel_list = ["All"] + sorted(gdf_quarters['name'].unique().tolist()) if not gdf_quarters.empty else ["All"]
    
    # --- Options Tab ---
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
        
        # Legend
        if layer_type == "Land Cover Classes":
            st.markdown("#### Legend")
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 12px;'>"
            for cls_id, label in CLASS_LABELS.items():
                if cls_id == 13: continue
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

    # --- Stats Tab ---
    with tab_stats:
        if selected_veedel != "All" and not gdf_quarters.empty:
            row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
            if not row.empty:
                st.markdown(f"#### {selected_veedel}")
                area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row.columns else 0
                area_ha = area_m2 / 10000
                # Using Shape_Area which is usually in m2
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
             
             # Chart based on gdf_quarters if available to sort by name
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

# --- Map Logic ---
with col_map:
    # 1. Simplified Basemap
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    
    # 2. Results: Districts (Stadtbezirke) - REMOVED Outline Layer
    if not gdf_boroughs.empty:
        folium.GeoJson(
            gdf_boroughs,
            name="Districts",
            style_function=lambda x: {'fillColor': 'none', 'color': '#333333', 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.0},
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Bezirk:'])
        ).add_to(m)
    
    # 3. Quarters (Veedel)
    if not gdf_quarters.empty:
        def style_fn(feature):
            name = feature['properties']['name']
            if selected_veedel != "All" and name == selected_veedel:
                 return {'fillColor': '#ffff00', 'color': 'black', 'weight': 3, 'fillOpacity': 0.2}
            
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

    # 4. Tiles Overlay
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
        if layer == 'raw': path = f"datasets/{DATASET_ID}/data/raw/{tile_name}.jp2"
        elif layer == 'mask': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_mask.tif"
        elif layer == 'ndvi': path = f"datasets/{DATASET_ID}/data/processed/{tile_name}_ndvi.tif"
        try:
            with fs.open(path, "rb") as f: return f.read()
        except: return None
    
    for tile_name in tiles_to_display:
        ftype = 'mask'
        if layer_type == 'NDVI': ftype = 'ndvi'
        elif layer_type == 'Raw Satellite (RGB)': ftype = 'raw'
        
        b = open_bytes(tile_name, ftype)
        if not b and ftype == 'ndvi':
             b = open_bytes(tile_name, 'raw') 
        
        if b:
            try:
                with rasterio.MemoryFile(b) as memfile:
                    with memfile.open() as src:
                        bounds = src.bounds
                        crs = src.crs
                        try:
                             from rasterio.crs import CRS
                             if not crs.is_epsg_code: crs = CRS.from_epsg(25832)
                             wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
                        except: continue

                        data = src.read()
                        image_data = None
                        opacity = 0.7
                        
                        if layer_type == "Segmentation Mask (Green Highlight)":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for c in [4, 5, 6, 7, 9, 10, 11, 12]: rgba[mask == c] = [0, 255, 0, 200]
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "Land Cover Classes":
                            mask = data[0]
                            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                            for cls, col in FLAIR_COLORS.items(): rgba[mask == cls] = col
                            image_data = rgba
                            opacity = 0.8
                        elif layer_type == "NDVI":
                            if src.count == 1:
                                ndvi = data[0].astype('float32') * 0.0001
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-1, vmax=1)(ndvi))
                            else:
                                r, nir = data[0].astype('float32'), data[3].astype('float32')
                                ndvi = (nir - r) / (nir + r + 1e-8)
                                image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-1, vmax=1)(ndvi))
                        elif layer_type == "Raw Satellite (RGB)":
                            if src.count >= 3:
                                rgb = np.dstack((data[0], data[1], data[2]))
                                p2, p98 = np.percentile(rgb, (2, 98))
                                image_data = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                                opacity = 1.0
                                
                        if image_data is not None:
                             folium.raster_layers.ImageOverlay(
                                image=image_data,
                                bounds=[[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]],
                                opacity=opacity,
                                name=f"{layer_type} - {tile_name}",
                                control=False
                             ).add_to(m)
            except: pass

    # 5. Map Click Logic (Zoom Fix)
    map_output = st_folium(m, width=None, height=700, key="main_map_hf", use_container_width=True, returned_objects=["last_object_clicked"])
    
    if map_output['last_object_clicked']:
        props = map_output['last_object_clicked'].get('properties')
        if props and 'name' in props:
            clicked_name = props['name']
            if clicked_name in veedel_list and clicked_name != st.session_state['selected_veedel']:
                st.session_state['selected_veedel_widget'] = clicked_name
                st.session_state['selected_veedel'] = clicked_name
                # MANUAL UPDATE
                if not gdf_quarters.empty:
                     match = gdf_quarters[gdf_quarters['name'] == clicked_name]
                     if not match.empty:
                         centroid = match.geometry.centroid.iloc[0]
                         st.session_state['map_center'] = [centroid.y, centroid.x]
                         st.session_state['map_zoom'] = 13
                st.rerun()
