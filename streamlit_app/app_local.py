import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import duckdb
import rasterio
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
import geopandas as gpd

# 1. Page Configuration
st.set_page_config(page_title="GreenCologne (Local)", layout="wide")
st.title("🌿 GreenCologne (Local Preview)")

# 2. Paths & Constants
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "stats" / "stats.parquet"
QUARTERS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
BOROUGHS_FILE = DATA_DIR / "boundaries" / "Stadtbezirke.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

# FLAIR-HUB Color Palette
FLAIR_COLORS = {
    1: [219, 14, 154, 255],     # Building
    2: [255, 0, 0, 255],        # Impervious
    3: [143, 85, 41, 255],      # Barren
    4: [0, 255, 0, 255],        # Grass
    5: [32, 105, 10, 255],      # Brush
    6: [90, 31, 10, 255],       # Agriculture
    7: [48, 25, 23, 255],       # Tree
    8: [13, 227, 237, 255],     # Water
    9: [2, 161, 9, 255],        # Herbaceous
    10: [136, 68, 145, 255],    # Shrub
    11: [55, 23, 40, 255],      # Moss
    12: [173, 201, 198, 255],   # Lichen
    13: [0, 0, 0, 0],           # Unknown
}

if not STATS_FILE.exists():
    st.error(f"Stats file not found at {STATS_FILE}. Run scripts/05_generate_stats.py first.")
    st.stop()

# 3. Data Loading Functions
@st.cache_data
def load_quarters_with_stats():
    # Load Boundaries
    if not QUARTERS_FILE.exists(): return None
    gdf = gpd.read_parquet(QUARTERS_FILE)
    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
    
    # Load Stats
    try:
        con = duckdb.connect()
        df_s = con.execute(f"SELECT * FROM '{STATS_FILE}'").df()
        con.close()
        
        # Merge
        if 'name' in gdf.columns and 'name' in df_s.columns:
            gdf = gdf.merge(df_s, on='name', how='left')
            if 'green_area_m2' in gdf.columns:
                gdf['green_area_m2'] = gdf['green_area_m2'].fillna(0)
    except Exception as e:
        st.error(f"Error loading stats: {e}")
    
    return gdf

@st.cache_data
def load_boroughs():
    if not BOROUGHS_FILE.exists(): return None
    gdf = gpd.read_parquet(BOROUGHS_FILE)
    if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
    if 'STB_NAME' in gdf.columns: gdf = gdf.rename(columns={'STB_NAME': 'name'})
    return gdf

@st.cache_data
def get_tile_to_veedel_mapping():
    if not TILES_METADATA_FILE.exists() or not QUARTERS_FILE.exists(): return {}
    tiles_df = pd.read_csv(TILES_METADATA_FILE)
    geometries = []
    for _, row in tiles_df.iterrows():
        e, n = row['Koordinatenursprung_East'], row['Koordinatenursprung_North']
        geometries.append(box(e, n, e + 1000, n + 1000))
    tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
    
    quarters_gdf = gpd.read_parquet(QUARTERS_FILE)
    if quarters_gdf.crs != "EPSG:25832": quarters_gdf = quarters_gdf.to_crs("EPSG:25832")
        
    joined = gpd.sjoin(tiles_gdf, quarters_gdf, how="inner", predicate="intersects")
    return joined.groupby('name')['Kachelname'].apply(list).to_dict()

# Load Data
gdf_quarters = load_quarters_with_stats()
gdf_boroughs = load_boroughs()
tile_mapping = get_tile_to_veedel_mapping()
# df_stats is now implied in gdf_quarters, but if we need a pure df for charts:
df_stats = pd.DataFrame(gdf_quarters.drop(columns='geometry')) if gdf_quarters is not None else pd.DataFrame()

# 4. State Management
if 'selected_veedel' not in st.session_state:
    st.session_state['selected_veedel'] = "All"
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [50.9375, 6.9603] # Cologne Center
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 10 # Whole city view

# 5. Layout# --- Layout ---
col_map, col_details = st.columns([0.65, 0.35], gap="medium")

with col_details:
    st.markdown("### GreenCologne Analysis")
    
    # Tabs for Options and Statistics
    tab_opts, tab_stats = st.tabs(["🛠️ Options", "📊 Statistics"])
    
    veedel_list = ["All"] + sorted(gdf_quarters['name'].unique().tolist()) if gdf_quarters is not None and not gdf_quarters.empty else ["All"]
    
    # --- Options Tab ---
    with tab_opts:
        # Veedel Selection
        def on_veedel_change():
            sel = st.session_state['selected_veedel_widget']
            st.session_state['selected_veedel'] = sel
            if sel == "All":
                st.session_state['map_center'] = [50.9375, 6.9603]
                st.session_state['map_zoom'] = 10
            elif gdf_quarters is not None and not gdf_quarters.empty:
                 match = gdf_quarters[gdf_quarters['name'] == sel]
                 if not match.empty:
                     centroid = match.geometry.centroid.iloc[0]
                     st.session_state['map_center'] = [centroid.y, centroid.x]
                     st.session_state['map_zoom'] = 13 # Zoom in closer

        selected_veedel = st.selectbox(
            "Select Quarter (Veedel/Stadtviertel):", 
            veedel_list, 
            key='selected_veedel_widget', 
            on_change=on_veedel_change,
            index=veedel_list.index(st.session_state['selected_veedel']) if st.session_state['selected_veedel'] in veedel_list else 0
        )

        # Tiles Logic
        current_veedel_tiles = []
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            # For local, check existence
            filtered_tiles = []
            for t in veedel_tiles:
                 if (DATA_DIR / "raw" / f"{t}.jp2").exists() or (PROCESSED_DIR / f"{t}_mask.tif").exists():
                     filtered_tiles.append(t)
            current_veedel_tiles = sorted(filtered_tiles)
        else:
            # "All" mode: Show ALL available tiles (could be many)
            # For local, scan directory? Or use mapping keys?
            # Let's take all from mapping for now but limit downstream
            all_t = set()
            for k, v in tile_mapping.items():
                all_t.update(v)
            current_veedel_tiles = sorted(list(all_t))

        tile_options = ["- Select a Tile -"] + current_veedel_tiles
        selected_tile = st.selectbox("Select Tile:", tile_options)
        if selected_tile == "- Select a Tile -": selected_tile = None

        # Layer Selection
        layer_type = st.radio(
            "Select Layer:",
            ["Segmentation Mask (Green Highlight)", "Land Cover Classes", "NDVI", "Raw Satellite (RGB)"],
            index=0
        )
        
        st.info("ℹ️ Select a specific Veedel or Tile to view satellite imagery.")

    # --- Stats Tab ---
    with tab_stats:
        if selected_veedel != "All" and gdf_quarters is not None and not gdf_quarters.empty:
            row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
            if not row.empty:
                st.markdown(f"#### {selected_veedel}")
                area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row else 0
                area_ha = area_m2 / 10000
                total_area_m2 = row['Shape_Area'].values[0] if 'Shape_Area' in row else 1
                pct = (area_m2 / total_area_m2) * 100
                
                col_metric1, col_metric2 = st.columns(2)
                col_metric1.metric("Green Area", f"{area_ha:.2f} ha")
                col_metric2.metric("Green Coverage", f"{pct:.1f}%")
                
                # Simple gauge chart using Plotly
                fig_gauge = px.pie(
                    names=['Green', 'Other'], 
                    values=[area_m2, total_area_m2 - area_m2],
                    color=['Green', 'Other'],
                    color_discrete_map={'Green': 'green', 'Other': 'lightgray'},
                    hole=0.6,
                    title="Green Coverage"
                )
                fig_gauge.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
        elif gdf_quarters is not None and not gdf_quarters.empty:
             st.markdown("#### City Overview")
             total_ha = gdf_quarters['green_area_m2'].sum() / 10000 if 'green_area_m2' in gdf_quarters.columns else 0
             st.metric("Total Green Area", f"{total_ha:.2f} ha")
             
             # Top 10 Chart
             top_10 = gdf_quarters.sort_values('green_area_m2', ascending=False).head(10)
             fig = px.bar(
                top_10, 
                x='name', 
                y='green_area_m2',
                title="Top 10 Greenest Quarters",
                labels={'green_area_m2': 'Green Area (m²)', 'name': ''},
                color='green_area_m2',
                color_continuous_scale='Greens'
            )
             fig.update_layout(height=400, margin=dict(l=0, r=0))
             st.plotly_chart(fig, use_container_width=True)

# --- Map Logic ---
with col_map:
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
    folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Satellite (Google)", overlay=False, control=True, show=False).add_to(m)

    # 1. Boroughs (Background)
    if gdf_boroughs is not None and not gdf_boroughs.empty:
        folium.GeoJson(
            gdf_boroughs,
            name="Results: Districts (Stadtbezirke)",
            style_function=lambda x: {'fillColor': 'none', 'color': '#333333', 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.0},
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Bezirk:'])
        ).add_to(m)
    
    # 2. Quarters (Interactive)
    if gdf_quarters is not None and not gdf_quarters.empty:
        # Style function handling both highlight and chloropleth
        def style_fn(feature):
            name = feature['properties']['name']
            
            # Selection Highlight
            if selected_veedel != "All" and name == selected_veedel:
                 return {'fillColor': '#ffff00', 'color': 'black', 'weight': 3, 'fillOpacity': 0.2}
            
            # Default Chloropleth (if no selection or not selected)
            green_area = feature['properties'].get('green_area_m2', 0)
            has_green = green_area > 0
            
            # If "All" is selected, we show chloropleth
            # If specific Veedel selected, we fade others
            opacity = 0.4 if selected_veedel == "All" else 0.1
            
            return {
                'fillColor': 'green' if has_green else 'gray',
                'color': '#666666',
                'weight': 1,
                'fillOpacity': 0.1 if selected_veedel != "All" else 0.4
            }
        
        folium.GeoJson(
            gdf_quarters,
            name="Quarters (Veedel)",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'green_area_m2'], 
                aliases=['Veedel:', 'Green Area (m²):'], 
                localize=True
            )
        ).add_to(m)

    # 3. City Outline (Fixed Dissolve)
    if gdf_boroughs is not None and not gdf_boroughs.empty:
         # Use a constant column to dissolve all into one polygon
         gdf_boroughs['dissolve_const'] = 1
         city_outline = gdf_boroughs.dissolve(by='dissolve_const')
         folium.GeoJson(
            city_outline,
            name="Cologne City Outline",
            style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2, 'dashArray': '10, 5', 'fillOpacity': 0.0}
         ).add_to(m)

    # 4. Tile Overlay
    tiles_to_display = []
    
    # Logic: 
    # - If tile selected -> Show that tile (Priority)
    # - If Veedel selected -> Show first 5 tiles of Veedel
    # - If "All" selected -> Show NOTHING by default (to avoid crash), unless user picks a tile
    
    if selected_tile:
        tiles_to_display = [selected_tile]
    elif selected_veedel != "All" and current_veedel_tiles:
         if len(current_veedel_tiles) > 5:
             st.toast(f"Displaying 5/{len(current_veedel_tiles)} tiles for performance. Select a specific tile to view others.", icon="ℹ️")
             tiles_to_display = current_veedel_tiles[:5]
         else:
             tiles_to_display = current_veedel_tiles
    
    # Processing Loop (unchanged logic, just re-indented)
    def read_local_raster(tile_name, layer):
        f_mask = PROCESSED_DIR / f"{tile_name}_mask.tif"
        f_ndvi = PROCESSED_DIR / f"{tile_name}_ndvi.tif"
        f_raw = DATA_DIR / "raw" / f"{tile_name}.jp2"
        target = None
        if layer == "NDVI": target = f_ndvi if f_ndvi.exists() else f_raw
        elif "Mask" in layer or "Classes" in layer: target = f_mask
        elif "Raw" in layer: target = f_raw
        if not target or not target.exists(): return None, None
        try: return rasterio.open(target), target
        except: return None, None

    for tile_name in tiles_to_display:
        src, path = read_local_raster(tile_name, layer_type)
        if src:
            with src:
                 bounds = src.bounds
                 from rasterio.crs import CRS
                 src_crs = src.crs
                 if not src_crs: src_crs = CRS.from_epsg(25832)
                 try: wgs_bounds = transform_bounds(src_crs, "EPSG:4326", *bounds)
                 except: continue

                 data = src.read()
                 image_data = None
                 opacity = 0.7
                 
                 if layer_type == "Segmentation Mask (Green Highlight)":
                        mask_data = data[0]
                        rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                        for c in [4, 5, 6, 7, 9, 10, 11, 12]: rgba[mask_data == c] = [0, 255, 0, 200]
                        image_data = rgba
                        opacity = 0.8
                 elif layer_type == "Land Cover Classes":
                        mask_data = data[0]
                        rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                        for cls_id, color in FLAIR_COLORS.items(): rgba[mask_data == cls_id] = color
                        image_data = rgba
                        opacity = 0.8
                 elif layer_type == "NDVI":
                         if src.count == 1:
                            ndvi = data[0].astype('float32') * 0.0001
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

    folium.LayerControl().add_to(m)
    
    # Optimized Return: prevent crash by limiting data transfer
    map_output = st_folium(m, width=None, height=700, key="main_map", use_container_width=True, returned_objects=["last_object_clicked"])
    
    if map_output['last_object_clicked']:
        props = map_output['last_object_clicked'].get('properties')
        if props and 'name' in props:
            clicked_name = props['name']
            if clicked_name in veedel_list and clicked_name != st.session_state['selected_veedel']:
                st.session_state['selected_veedel_widget'] = clicked_name
                st.session_state['selected_veedel'] = clicked_name
                st.rerun()
