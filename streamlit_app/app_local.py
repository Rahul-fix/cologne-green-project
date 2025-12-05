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
# 2. Paths & Constants
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "stats" / "extended_stats.parquet"
QUARTERS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
BOROUGHS_FILE = DATA_DIR / "boundaries" / "Stadtbezirke.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

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

# 4. State Management
if 'selected_veedel' not in st.session_state:
    st.session_state['selected_veedel'] = "All"
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [50.9375, 6.9603] # Cologne Center
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 10 # Whole city view

# --- Layout ---
col_map, col_details = st.columns([0.65, 0.35], gap="medium")

with col_details:
    st.markdown("### GreenCologne Analysis")
    tab_opts, tab_stats = st.tabs(["🛠️ Options", "📊 Statistics"])
    
    veedel_list = ["All"] + sorted(gdf_quarters['name'].unique().tolist()) if gdf_quarters is not None and not gdf_quarters.empty else ["All"]
    
    with tab_opts:
        def update_zoom_for_veedel(veedel_name):
             if veedel_name == "All":
                st.session_state['map_center'] = [50.9375, 6.9603]
                st.session_state['map_zoom'] = 10
             elif gdf_quarters is not None:
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

        # Tiles Logic
        current_veedel_tiles = []
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            filtered_tiles = []
            for t in veedel_tiles:
                 if (DATA_DIR / "raw" / f"{t}.jp2").exists() or (PROCESSED_DIR / f"{t}_mask.tif").exists():
                     filtered_tiles.append(t)
            current_veedel_tiles = sorted(filtered_tiles)
        else:
            all_t = set()
            for k, v in tile_mapping.items(): all_t.update(v)
            current_veedel_tiles = sorted(list(all_t))

        tile_options = ["- Select a Tile -"] + current_veedel_tiles
        selected_tile = st.selectbox("Select Tile:", tile_options)
        if selected_tile == "- Select a Tile -": selected_tile = None

        st.info("ℹ️ Select a specific Veedel or Tile to view satellite imagery.")
        
        # Layer Selection (Reverted to Radio)
        layer_selection = st.radio(
            "Select Layer:",
            ["Raw Satellite (RGB)", "Segmentation Mask (Green Highlight)", "Land Cover Classes", "NDVI"],
            index=1,
            horizontal=True
        )
        
        # Legend (Conditional based on layer)
        if layer_selection == "Land Cover Classes":
            st.markdown("#### Legend")
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 12px;'>"
            for cls_id, label in CLASS_LABELS.items():
                if cls_id == 13: continue
                c = FLAIR_COLORS[cls_id]
                color_css = f"rgba({c[0]},{c[1]},{c[2]},{c[3]/255})"
                legend_html += f"<div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: {color_css}; margin-right: 5px; border: 1px solid #ccc;'></div>{label}</div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

    with tab_stats:
        if selected_veedel != "All" and gdf_quarters is not None and not gdf_quarters.empty:
            row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
            if not row.empty:
                st.markdown(f"#### {selected_veedel} Stats")
                area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row else 0
                total_area_m2 = row['Shape_Area'].values[0] if 'Shape_Area' in row else 1
                
                c1, c2 = st.columns(2)
                c1.metric("Green Area", f"{(area_m2/10000):.2f} ha")
                c2.metric("Green Coverage", f"{(area_m2/total_area_m2)*100:.1f}%")
                
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

# --- Map Logic ---
with col_map:
    # 1. Simplified Basemap (CartoDB only)
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    
    # 2. Results: Districts
    if gdf_boroughs is not None and not gdf_boroughs.empty:
        folium.GeoJson(
            gdf_boroughs,
            name="Districts",
            style_function=lambda x: {'fillColor': 'none', 'color': '#333333', 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.0},
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Bezirk:'])
        ).add_to(m)
    
    # 3. Quarters (Veedel)
    if gdf_quarters is not None and not gdf_quarters.empty:
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
         tiles_to_display = current_veedel_tiles[:5] # Limit 5
    
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

    # Only add the SELECTED layer
    layer_type = layer_selection
    
    # Create a FeatureGroup for the single selected layer
    # We still use FeatureGroup so we can control it via LayerControl if we wanted, 
    # but here we rely on st.radio reload.
    fg = folium.FeatureGroup(name=layer_type, show=True)
    
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
                        ).add_to(fg)
    
    fg.add_to(m)

    # LayerControl is optional if we only have 1 dynamic layer + standard overlays,
    # but good to keep for switching off Veedel/Districts if desired.
    folium.LayerControl().add_to(m)

    # 5. Map Click Logic (Zoom Fix)
    map_output = st_folium(m, width=None, height=700, key="main_map", use_container_width=True, returned_objects=["last_object_clicked"])
    
    if map_output['last_object_clicked']:
        props = map_output['last_object_clicked'].get('properties')
        if props and 'name' in props:
            clicked_name = props['name']
            if clicked_name in veedel_list and clicked_name != st.session_state['selected_veedel']:
                st.session_state['selected_veedel_widget'] = clicked_name
                st.session_state['selected_veedel'] = clicked_name
                # MANUAL UPDATE with centroid fix
                if gdf_quarters is not None:
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
