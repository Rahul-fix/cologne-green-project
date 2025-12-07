import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point

# Import extracted utilities
from utils import (
    DATA_DIR, PROCESSED_DIR, FLAIR_COLORS, CLASS_LABELS,
    load_quarters_with_stats, load_boroughs, get_tile_to_veedel_mapping,
    get_mosaic_data_local
)

# 1. Page Configuration
st.set_page_config(page_title="GreenCologne (Local)", layout="wide")
st.title("🌿 GreenCologne (Local Preview)")

# 2. Data Loading
gdf_quarters = load_quarters_with_stats()
gdf_boroughs = load_boroughs()
tile_mapping = get_tile_to_veedel_mapping()

# 3. State Management
if 'selected_veedel' not in st.session_state: st.session_state['selected_veedel'] = "All"
if 'map_center' not in st.session_state: st.session_state['map_center'] = [50.9375, 6.9603] # Cologne
if 'map_zoom' not in st.session_state: st.session_state['map_zoom'] = 11
if 'map_click_counter' not in st.session_state: st.session_state['map_click_counter'] = 0

# --- Helper Functions ---
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

# --- Layout ---
col_map, col_details = st.columns([0.65, 0.35], gap="medium")

with col_details:
    st.markdown("### GreenCologne Analysis")
    tab_opts, tab_stats = st.tabs(["🛠️ Options", "📊 Statistics"])
    
    veedel_list = ["All"] + sorted(gdf_quarters['name'].unique().tolist()) if gdf_quarters is not None and not gdf_quarters.empty else ["All"]
    
    # --- Tab 1: Options ---
    with tab_opts:
        # Sync Widget
        if 'selected_veedel_widget' in st.session_state and st.session_state['selected_veedel'] != st.session_state['selected_veedel_widget']:
            st.session_state['selected_veedel_widget'] = st.session_state['selected_veedel']

        selected_veedel = st.selectbox(
            "Select Quarter (Veedel/Stadtviertel):", 
            veedel_list, 
            key='selected_veedel_widget', 
            on_change=on_veedel_change,
            index=veedel_list.index(st.session_state['selected_veedel']) if st.session_state['selected_veedel'] in veedel_list else 0
        )

        # Tile Logic - Only load tiles when a specific veedel is selected
        tiles_to_display = []
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            filtered_tiles = []
            for t in veedel_tiles:
                 try:
                     if (DATA_DIR / "raw" / f"{t}.jp2").exists() or (PROCESSED_DIR / f"{t}_mask.tif").exists():
                         filtered_tiles.append(t)
                 except: pass
            tiles_to_display = sorted(filtered_tiles)
        
        # Layer Selection
        layer_selection = st.radio(
            "Select Layer:",
            ["Satellite", "Land Cover", "NDVI"],
            index=2,
            horizontal=True
        )
        
        # Legends
        st.markdown("#### Legends")
        st.markdown("**Veedel Health (Mean NDVI)**")
        st.markdown("""
        <div style="background: linear-gradient(to right, #d73027, #ffffbf, #1a9850); height: 10px; width: 100%; border-radius: 5px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px;"><span>0.0 (Low)</span><span>0.3</span><span>0.6+ (High)</span></div>
        <div style="font-size: 11px; color: #666; margin-bottom: 15px;">*Average vegetation index per district.</div>
        """, unsafe_allow_html=True)

        if layer_selection == "Land Cover":
            st.markdown("**Land Cover Classes**")
            legend_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 12px;'>"
            for cls_id, label in CLASS_LABELS.items():
                if cls_id == 13: continue
                c = FLAIR_COLORS[cls_id]
                legend_html += f"<div style='display: flex; align-items: center;'><div style='width: 12px; height: 12px; background: rgba({c[0]},{c[1]},{c[2]},{c[3]/255}); margin-right: 5px; border: 1px solid #ccc;'></div>{label}</div>"
            st.markdown(legend_html + "</div>", unsafe_allow_html=True)
            
        elif layer_selection == "NDVI":
            st.markdown("**Pixel Vegetation Index (NDVI)**")
            st.markdown("""
            <div style="background: linear-gradient(to right, #d73027, #ffffbf, #1a9850); height: 10px; width: 100%; border-radius: 5px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px;"><span>-0.4 (Water)</span><span>0.3</span><span>1.0 (Dense)</span></div>
            """, unsafe_allow_html=True)

    # --- Tab 2: Statistics ---
    with tab_stats:
        if gdf_quarters is not None and not gdf_quarters.empty:
            title = "Cologne (All Veedels) Stats"
            area_m2 = 0
            total_area_m2 = 1
            class_data_source = None

            if selected_veedel != "All":
                row = gdf_quarters[gdf_quarters['name'] == selected_veedel]
                if not row.empty:
                    title = f"{selected_veedel} Stats"
                    area_m2 = row['green_area_m2'].values[0] if 'green_area_m2' in row else 0
                    total_area_m2 = row['Shape_Area'].values[0] if 'Shape_Area' in row else 1
                    class_data_source = row
                else: 
                     st.warning(f"No stats for {selected_veedel}")
            else:
                area_m2 = gdf_quarters['green_area_m2'].sum() if 'green_area_m2' in gdf_quarters else 0
                total_area_m2 = gdf_quarters['Shape_Area'].sum() if 'Shape_Area' in gdf_quarters else 1
                class_cols = [c for c in gdf_quarters.columns if str(c).startswith('area_')]
                if class_cols:
                    class_data_source = pd.DataFrame([{c: gdf_quarters[c].sum() for c in class_cols}])

            st.markdown(f"#### {title}")
            c1, c2 = st.columns(2)
            c1.metric("Green Area", f"{(area_m2/10000):.2f} ha")
            c2.metric("Green Coverage", f"{(area_m2/total_area_m2)*100:.1f}%")
            st.divider()
            
            if class_data_source is not None and not class_data_source.empty:
                class_cols = [c for c in class_data_source.columns if str(c).startswith('area_')]
                if class_cols:
                    class_data = class_data_source[class_cols].T.reset_index()
                    class_data.columns = ['class_col', 'area_m2']
                    class_data['class_id'] = pd.to_numeric(class_data['class_col'].str.replace('area_', '', regex=False), errors='coerce').fillna(0).astype(int)
                    class_data['class_name'] = class_data['class_id'].map(CLASS_LABELS)
                    class_data['color'] = class_data['class_id'].map(lambda x: f"rgba({FLAIR_COLORS[x][0]},{FLAIR_COLORS[x][1]},{FLAIR_COLORS[x][2]}, 1)")
                    class_data = class_data.sort_values(by='area_m2', ascending=False)
                    
                    fig_bar = px.bar(
                        class_data, x='class_name', y='area_m2', 
                        title=f"Land Cover Distribution", 
                        labels={'area_m2': 'Area (m²)'}, color='class_name', 
                        color_discrete_map={row['class_name']: row['color'] for _, row in class_data.iterrows()}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else: st.info("No detailed land cover data.")

# --- Map View ---
with col_map:
    # 1. Base Map
    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron", crs='EPSG3857')
    
    # 2. Results: Districts
    if gdf_boroughs is not None and not gdf_boroughs.empty:
        folium.GeoJson(
            gdf_boroughs, name="Districts",
            style_function=lambda x: {'fillColor': 'none', 'color': '#333333', 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.0},
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Bezirk:'])
        ).add_to(m)
    
    # 3. Quarters (Veedel)
    if gdf_quarters is not None and not gdf_quarters.empty:
        min_ndvi = gdf_quarters['ndvi_mean'].min() if 'ndvi_mean' in gdf_quarters else 0
        max_ndvi = gdf_quarters['ndvi_mean'].max() if 'ndvi_mean' in gdf_quarters else 0.6
        if pd.isna(min_ndvi): min_ndvi = 0
        if pd.isna(max_ndvi): max_ndvi = 0.6
        
        def get_style(feature):
            name = feature['properties']['name']
            is_sel = (selected_veedel != "All" and name == selected_veedel)
            val = feature['properties'].get('ndvi_mean')
            
            fill_color = 'gray'
            if is_sel: fill_color = '#ffff00'
            elif val is not None and not pd.isna(val):
                norm = max(0, min(1, (val - min_ndvi) / (max_ndvi - min_ndvi + 1e-9)))
                fill_color = mcolors.to_hex(plt.get_cmap('RdYlGn')(norm))
            
            return {
                'fillColor': fill_color,
                'color': 'black' if is_sel else '#666666',
                'weight': 3 if is_sel else 1,
                'fillOpacity': 0.0 if is_sel else 0.6 
            }

        folium.GeoJson(
            gdf_quarters, name="Veedel (NDVI)", style_function=get_style,
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'green_area_m2', 'green_pct', 'ndvi_mean'], 
                aliases=['Veedel:', 'Green Area (m²):', 'Green Coverage (%):', 'Mean NDVI:'], 
                localize=True, fmt='.2f'
            )
        ).add_to(m)

    # 4. Tiles Grid (Mosaic)
    if tiles_to_display:
        with st.spinner(f"Loading {len(tiles_to_display)} tiles..."):
            mosaic_img, mosaic_bounds = get_mosaic_data_local(tiles_to_display, layer_selection)
            if mosaic_img is not None and mosaic_bounds:
                folium.raster_layers.ImageOverlay(
                    image=mosaic_img, bounds=mosaic_bounds,
                    opacity=0.8 if "Land Cover" in layer_selection else 1.0,
                    name=f"Mosaic - {layer_selection}", control=False
                ).add_to(m)

    folium.LayerControl().add_to(m)

    # 5. Click Logic (Hybrid)
    map_key = f"map_{st.session_state['selected_veedel']}_{st.session_state['map_zoom']}_{st.session_state['map_click_counter']}"
    map_output = st_folium(m, width=None, height=700, key=map_key, use_container_width=True, returned_objects=["last_object_clicked", "last_clicked"])
    
    clicked_name_final = None
    if map_output:
        # A: Check Object Property
        if map_output.get('last_object_clicked'):
            props = map_output['last_object_clicked'].get('properties', {})
            if props and 'name' in props: clicked_name_final = props['name']
        
        # B: Spatial Query Fallback
        if not clicked_name_final and map_output.get('last_clicked'):
             lat = map_output['last_clicked']['lat']
             lng = map_output['last_clicked']['lng']
             if lat and lng and gdf_quarters is not None:
                 p = Point(lng, lat)
                 matches = gdf_quarters[gdf_quarters.geometry.contains(p)]
                 if not matches.empty: clicked_name_final = matches['name'].iloc[0]

    if clicked_name_final and clicked_name_final in veedel_list and clicked_name_final != st.session_state['selected_veedel']:
        st.session_state['selected_veedel'] = clicked_name_final
        update_zoom_for_veedel(clicked_name_final)
        st.session_state['map_click_counter'] += 1
        st.rerun()
