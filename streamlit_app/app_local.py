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
# FLAIR-HUB Color Palette & Labels (Matched to flair-hub-qgis-style-cosia-num.qml)
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

@st.cache_data(show_spinner=False)
def get_image_layer(tile_id):
    """
    Load visualization layer.
    Prioritizes 'web_optimized' (small) tiles for performance.
    Fallback to 'raw' (large) if optimized not found.
    """
    optimized_path = Path(f"data/web_optimized/{tile_id}.tif")
    raw_path = Path(f"data/raw/{tile_id}.jp2")
    
    # Try Optimized first
    if optimized_path.exists():
        try:
            with rasterio.open(optimized_path) as src:
                # Optimized tiles are usually 8-bit RGB
                img = src.read([1, 2, 3])
                bounds = src.bounds
                image = np.moveaxis(img, 0, -1)
                return image, [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        except Exception as e:
            pass

    # Fallback to Raw
    if raw_path.exists():
        try:
            with rasterio.open(raw_path) as src:
                img = src.read([1, 2, 3])
                bounds = src.bounds
                image = np.moveaxis(img, 0, -1)
                if image.dtype == 'uint16':
                     image = (image / 256).astype('uint8')
                return image, [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        except Exception as e:
             return None, None
             
    return None, None

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
        image_data = None
        opacity = 0.7
        wgs_bounds = None
        
        # 1. Optimized RGB Layer
        if layer_type == "Raw Satellite (RGB)":
            img, bounds = get_image_layer(tile_name)
            if img is not None:
                image_data = img
                opacity = 1.0
                try:
                    from rasterio.crs import CRS
                    wgs_bounds = transform_bounds(CRS.from_epsg(25832), "EPSG:4326", 
                                                bounds[0][1], bounds[0][0], bounds[1][1], bounds[1][0])
                except: continue

        # 2. Optimized Mask or NDVI (NEW)
        elif "Mask" in layer_type or "Classes" in layer_type or layer_type == "NDVI":
            # Check for optimized file first
            suffix = "_mask" if ("Mask" in layer_type or "Classes" in layer_type) else "_ndvi"
            opt_path = DATA_DIR / "web_optimized" / f"{tile_name}{suffix}.tif"
            
            src = None
            if opt_path.exists():
                try: src = rasterio.open(opt_path)
                except: pass
            
            # Fallback to Processed/Raw if Optimized missing
            if not src:
                 src, path = read_local_raster(tile_name, layer_type)
            
            if src:
                with src:
                    bounds = src.bounds
                    crs = src.crs or rasterio.crs.CRS.from_epsg(25832)
                    try: 
                        wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
                    except: continue
                    
                    data = src.read()
                    
                    if layer_type == "Segmentation Mask (Green Highlight)":
                        mask_data = data[0]
                        rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                        # Veg classes (QML): 8=Herbaceous, 9=Agri, 10=Plowed, 11=Vineyard, 12=Deciduous, 13=Coniferous, 14=Brushwood
                        for c in [8, 9, 10, 11, 12, 13, 14]: rgba[mask_data == c] = [0, 255, 0, 200]
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
                            ndvi = data[0]
                            # Handle optimized float32 OR computed float32
                            # If optimized, it's already NDVI.
                            # If quantised? We kept it float32.
                            image_data = plt.get_cmap('RdYlGn')(mcolors.Normalize(vmin=-0.2, vmax=1)(ndvi))
                            # Note: plt returns (M, N, 4) float64 usually. Folium handles it?
                            # Optimisation: Convert to uint8?
                            # Folium/Base64 encoding handles floats but slower.
                            # But works.

        # 3. Add to Map
        if image_data is not None and wgs_bounds:
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
