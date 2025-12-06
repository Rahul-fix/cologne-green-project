import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import duckdb
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium
from shapely.geometry import box, Point
import geopandas as gpd

# 1. Page Configuration
st.set_page_config(page_title="GreenCologne (Local)", layout="wide")
st.title("🌿 GreenCologne (Local Preview)")

# 2. Paths & Constants
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "stats" / "extended_stats.parquet"
QUARTERS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
BOROUGHS_FILE = DATA_DIR / "boundaries" / "Stadtbezirke.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

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
    st.session_state['map_zoom'] = 11 # Whole city view
if 'map_click_counter' not in st.session_state:
    st.session_state['map_click_counter'] = 0

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
                     st.session_state['map_zoom'] = 14

        def on_veedel_change():
            sel = st.session_state['selected_veedel_widget']
            st.session_state['selected_veedel'] = sel
            update_zoom_for_veedel(sel)

        # Sync Widget with Session State (Map Click)
        if 'selected_veedel_widget' in st.session_state and st.session_state['selected_veedel'] != st.session_state['selected_veedel_widget']:
            st.session_state['selected_veedel_widget'] = st.session_state['selected_veedel']

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

        # Tile Selection (Hidden/Automatic)
        tiles_to_display = current_veedel_tiles
        selected_tile = None
        
        # Layer Selection
        layer_selection = st.radio(
            "Select Layer:",
            ["Satellite", "Land Cover", "NDVI"],
            index=1,
            horizontal=True
        )
        
        # --- LEGENDS ---
        st.markdown("#### Legends")
        
        # 1. District Mean NDVI Legend (Always visible)
        st.markdown("**Veedel Health (Mean NDVI)**")
        legend_html_veedel = """
        <div style="background: linear-gradient(to right, #d73027, #ffffbf, #1a9850); height: 10px; width: 100%; border-radius: 5px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px;">
            <span>0.0 (Low)</span>
            <span>0.3</span>
            <span>0.6+ (High)</span>
        </div>
        <div style="font-size: 11px; color: #666; margin-bottom: 15px;">
            *Average vegetation index per district.
        </div>
        """
        st.markdown(legend_html_veedel, unsafe_allow_html=True)

        # 2. Layer Specific Legends
        if layer_selection == "Land Cover":
            st.markdown("**Land Cover Classes**")
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 12px;'>"
            for cls_id, label in CLASS_LABELS.items():
                if cls_id == 13: continue
                c = FLAIR_COLORS[cls_id]
                color_css = f"rgba({c[0]},{c[1]},{c[2]},{c[3]/255})"
                legend_html += f"<div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: {color_css}; margin-right: 5px; border: 1px solid #ccc;'></div>{label}</div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
            
        elif layer_selection == "NDVI":
            st.markdown("**Pixel Vegetation Index (NDVI)**")
            legend_html_ndvi = """
            <div style="background: linear-gradient(to right, #d73027, #ffffbf, #1a9850); height: 10px; width: 100%; border-radius: 5px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px;">
                <span>-0.4 (Water)</span>
                <span>0.3</span>
                <span>1.0 (Dense)</span>
            </div>
            """
            st.markdown(legend_html_ndvi, unsafe_allow_html=True)
            
# --- Map Logic ---
    with tab_stats:
        if gdf_quarters is not None and not gdf_quarters.empty:
            
            # Determine data scope (Specific Veedel vs All)
            class_data_source = None
            title = ""
            area_m2 = 0
            total_area_m2 = 1

            if selected_veedel != "All":
                row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
                if row.empty:
                    st.warning(f"No data found for {selected_veedel}")
                else:
                    title = f"{selected_veedel} Stats"
                    # Single Veedel Data
                    area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row else 0
                    total_area_m2 = row['Shape_Area'].values[0] if 'Shape_Area' in row else 1
                    class_data_source = row
            else:
                title = "Cologne (All Veedels) Stats"
                # Aggregate Data
                area_m2 = gdf_quarters['green_area_m2'].sum() if 'green_area_m2' in gdf_quarters else 0
                total_area_m2 = gdf_quarters['Shape_Area'].sum() if 'Shape_Area' in gdf_quarters else 1
                # Aggregate Class Data (Sum all area_ columns)
                class_cols = [c for c in gdf_quarters.columns if str(c).startswith('area_')]
                if class_cols:
                    # Create a dummy row dict with summed values
                    summed_data = {c: gdf_quarters[c].sum() for c in class_cols}
                    # Need to reshape it to match row structure (DataFrame with index match)
                    class_data_source = pd.DataFrame([summed_data])
                else:
                    class_data_source = pd.DataFrame()

            # Display Stats
            st.markdown(f"#### {title}")
            
            c1, c2 = st.columns(2)
            c1.metric("Green Area", f"{(area_m2/10000):.2f} ha")
            c2.metric("Green Coverage", f"{(area_m2/total_area_m2)*100:.1f}%")
            
            st.divider()
            
            # Class Breakdown Chart
            if class_data_source is not None and not class_data_source.empty:
                class_cols = [c for c in class_data_source.columns if str(c).startswith('area_')]
                if class_cols:
                    class_data = class_data_source[class_cols].T.reset_index()
                    class_data.columns = ['class_col', 'area_m2']
                    # Parse class_id from 'area_X'
                    class_data['class_id'] = class_data['class_col'].str.replace('area_', '', regex=False)
                    class_data['class_id'] = pd.to_numeric(class_data['class_id'], errors='coerce')
                    class_data = class_data.dropna(subset=['class_id'])
                    class_data['class_id'] = class_data['class_id'].astype(int)
                    
                    class_data['class_name'] = class_data['class_id'].map(CLASS_LABELS)
                    class_data['color'] = class_data['class_id'].map(lambda x: f"rgba({FLAIR_COLORS[x][0]},{FLAIR_COLORS[x][1]},{FLAIR_COLORS[x][2]}, 1)")
                    
                    # Sort by area
                    class_data = class_data.sort_values(by='area_m2', ascending=False)
                    
                    fig_bar = px.bar(
                        class_data, x='class_name', y='area_m2', 
                        title=f"Land Cover Distribution ({'Total' if selected_veedel == 'All' else selected_veedel})", 
                        labels={'area_m2': 'Area (m²)', 'class_name': 'Class'},
                        color='class_name', 
                        color_discrete_map={row['class_name']: row['color'] for _, row in class_data.iterrows()}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No detailed land cover statistics available.")
with col_map:
    # Use standard 3857 Map to ensure compatibility with standard basemaps (CartoDB)
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron", crs='EPSG3857')
    
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
        min_ndvi = gdf_quarters['ndvi_mean'].min() if 'ndvi_mean' in gdf_quarters else 0
        max_ndvi = gdf_quarters['ndvi_mean'].max() if 'ndvi_mean' in gdf_quarters else 0.6
        if pd.isna(min_ndvi): min_ndvi = 0
        if pd.isna(max_ndvi): max_ndvi = 0.6
        
        def get_color(feature):
            if selected_veedel != "All" and feature['properties']['name'] == selected_veedel:
                return '#ffff00'
            val = feature['properties'].get('ndvi_mean')
            if val is None or pd.isna(val): return 'gray'
            norm = (val - min_ndvi) / (max_ndvi - min_ndvi + 1e-9)
            norm = max(0, min(1, norm))
            rgba = plt.get_cmap('RdYlGn')(norm)
            return mcolors.to_hex(rgba)

        def style_fn(feature):
            name = feature['properties']['name']
            is_selected = (selected_veedel != "All" and name == selected_veedel)
            return {
                'fillColor': get_color(feature),
                'color': 'black' if is_selected else '#666666',
                'weight': 3 if is_selected else 1,
                'fillOpacity': 0.0 if is_selected else 0.6 
            }
        
        # Ensure PCT exists
        if "green_area_m2" in gdf_quarters.columns and "Shape_Area" in gdf_quarters.columns:
             gdf_quarters["green_pct"] = (gdf_quarters["green_area_m2"] / gdf_quarters["Shape_Area"]) * 100
        else:
             gdf_quarters["green_pct"] = 0.0

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

    # 4. Tiles Overlay (Mosaicking)
    # helper for mosaic
    def get_mosaic_data(tile_names, layer_type):
        """
        Loads all tile paths, Mosaics them in native 25832, 
        Reprojects the SINGLE mosaic to WGS84 with Alpha,
        Colorizes if needed,
        Returns (image_data, wgs84_bounds)
        """
        sources = []
        try:
            for tile_name in tile_names:
                # Determine Suffix based on UPDATED layer names: ["Satellite", "Land Cover", "NDVI"]
                suffix = "_mask" if ("Land Cover" in layer_type) else "_ndvi"
                if layer_type == "Satellite": suffix = ""
                
                # Priority: Optimized -> Processed -> Raw
                # But we just need a path for rasterio.open
                opt_path = DATA_DIR / "web_optimized" / f"{tile_name}{suffix}.tif" # or just .tif for RGB
                raw_path = DATA_DIR / "raw" / f"{tile_name}.jp2"
                processed_mask = PROCESSED_DIR / f"{tile_name}_mask.tif"
                processed_ndvi = PROCESSED_DIR / f"{tile_name}_ndvi.tif"
                
                path_to_open = None
                if layer_type == "Satellite":
                    if opt_path.exists(): path_to_open = opt_path
                    elif raw_path.exists(): path_to_open = raw_path
                elif "Land Cover" in layer_type:
                    if opt_path.exists(): path_to_open = opt_path
                    elif processed_mask.exists(): path_to_open = processed_mask
                elif layer_type == "NDVI":
                    if opt_path.exists(): path_to_open = opt_path
                    elif processed_ndvi.exists(): path_to_open = processed_ndvi
                
                if path_to_open:
                    sources.append(rasterio.open(path_to_open))
            
            if not sources: return None, None

            # 1. Mosaic (Native CRS)
            # This merges them seamlessly in 25832 space
            mosaic, out_trans = merge(sources)
            
            # Close sources
            for s in sources: s.close()
            
            # 2. Prepare Reprojection to WGS84 (for Alpha/Rotation)
            # Source
            src_crs = CRS.from_epsg(25832)
            src_height, src_width = mosaic.shape[1], mosaic.shape[2]
            
            # Destination (Template)
            dst_crs = CRS.from_epsg(4326)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, 
                *rasterio.transform.array_bounds(src_height, src_width, out_trans)
            )
            
            # Allocate Dest Array (Bands, H, W)
            count = mosaic.shape[0]
            if layer_type == "Satellite" and count < 3: count = 1 
            
            dst_array = np.zeros((count, dst_height, dst_width), dtype=mosaic.dtype)
            
            reproject(
                source=mosaic,
                destination=dst_array,
                src_transform=out_trans,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            
            # Calculate WGS84 Bounds
            dst_bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
            # (w, s, e, n) -> [[s, w], [n, e]]
            folium_bounds = [[dst_bounds[1], dst_bounds[0]], [dst_bounds[3], dst_bounds[2]]]
            
            # 3. Post-Process for Visualization (Coloring & Permute)
            final_image = None
            
            if layer_type == "Satellite":
                if dst_array.shape[0] >= 3:
                    # (3, H, W) -> (H, W, 3)
                    rgb = np.moveaxis(dst_array[:3], 0, -1)
                    # Normalize if uint16
                    if rgb.dtype == 'uint16':
                        # Simple Min/Max Scale to avoid percentile overhead on large merges
                        p2, p98 = np.percentile(rgb[rgb > 0], (2, 98))
                        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                        final_image = (rgb * 255).astype(np.uint8)
                    else:
                        final_image = rgb
                    
                    # Alpha channel for rotation (where 0 is nodata)
                    alpha = np.any(final_image > 0, axis=2).astype(np.uint8) * 255
                    final_image = np.dstack((final_image, alpha))

            elif layer_type == "Land Cover":
                mask_data = dst_array[0] # (H, W)
                rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                for cls_id, color in FLAIR_COLORS.items(): 
                    rgba[mask_data == cls_id] = color
                final_image = rgba

            elif layer_type == "NDVI":
                ndvi = dst_array[0].astype('float32') # (H,W)
                # Normalize logic [-0.2, 1.0] -> [0, 1]
                norm = mcolors.Normalize(vmin=-0.4, vmax=1, clip=True)(ndvi)
                cmap = plt.get_cmap('RdYlGn')
                final_image_float = cmap(norm) # (H,W,4) float64
                final_image = (final_image_float * 255).astype(np.uint8)
                pass

            return final_image, folium_bounds

        except Exception as e:
            # st.error(f"Mosaic Error: {e}")
            return None, None

    # Execute Mosaic
    layer_type = layer_selection
    if tiles_to_display:
        with st.spinner(f"Merging & Warping {len(tiles_to_display)} tiles..."):
            mosaic_img, mosaic_bounds = get_mosaic_data(tiles_to_display, layer_type)
            
            if mosaic_img is not None and mosaic_bounds:
                folium.raster_layers.ImageOverlay(
                    image=mosaic_img,
                    bounds=mosaic_bounds,
                    opacity=0.8 if "Land Cover" in layer_type else 1.0,
                    name=f"Mosaic - {layer_type}",
                    control=False
                ).add_to(m)

    folium.LayerControl().add_to(m)

    # 5. Map Click Logic (Hybrid: Object + Spatial Fallback)
    if 'map_click_counter' not in st.session_state: st.session_state['map_click_counter'] = 0
    
    map_key = f"map_{st.session_state['selected_veedel']}_{st.session_state['map_zoom']}_{st.session_state['map_click_counter']}"
    
    # Request both object and coordinate data
    map_output = st_folium(m, width=None, height=700, key=map_key, use_container_width=True, returned_objects=["last_object_clicked", "last_clicked"])
    
    clicked_name_final = None

    if map_output:
        # Strategy A: Object Property (Fast, matches tooltip)
        if map_output.get('last_object_clicked'):
            obj = map_output['last_object_clicked']
            props = obj.get('properties', {})
            if props and 'name' in props:
                clicked_name_final = props['name']
        
        # Strategy B: Spatial Query Fallback (Robust)
        if not clicked_name_final and map_output.get('last_clicked'):
             lat = map_output['last_clicked']['lat']
             lng = map_output['last_clicked']['lng']
             if lat is not None and lng is not None:
                 p = Point(lng, lat)
                 if gdf_quarters is not None and not gdf_quarters.empty:
                     matches = gdf_quarters[gdf_quarters.geometry.contains(p)]
                     if not matches.empty:
                         clicked_name_final = matches['name'].iloc[0]

    # Execute Update
    if clicked_name_final and clicked_name_final in veedel_list and clicked_name_final != st.session_state['selected_veedel']:
        st.session_state['selected_veedel'] = clicked_name_final
        update_zoom_for_veedel(clicked_name_final)
        st.session_state['map_click_counter'] += 1
        st.rerun()
