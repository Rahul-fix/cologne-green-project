import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, box
import geopandas as gpd
import duckdb
import rasterio
from huggingface_hub import HfFileSystem
import shapely.wkb
from dotenv import load_dotenv
import os
from pathlib import Path

# Import extracted utilities (shared with app_local.py)
from utils import (
    FLAIR_COLORS, CLASS_LABELS, process_mosaic
)

# --- Cloud Configuration ---
load_dotenv()
env_path = Path(__file__).parent.parent / "DL_cologne_green" / ".env"
if env_path.exists():
    load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")
BASE_URL = f"hf://datasets/{DATASET_ID}"
STORAGE_OPTS = {"token": HF_TOKEN} if HF_TOKEN else None

# --- DuckDB Connection ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if HF_TOKEN:
        fs = HfFileSystem(token=HF_TOKEN)
        con.register_filesystem(fs)
    return con

con = get_db_connection()

# --- Cloud Data Loading Functions (mirror utils.py structure) ---
def safe_load_wkb(x):
    try: return shapely.wkb.loads(bytes(x))
    except: return None

@st.cache_data(ttl=3600)
def load_quarters_with_stats():
    """Load Veedel boundaries with stats - Cloud version of utils.load_quarters_with_stats"""
    try:
        query = f"""
            SELECT 
                v.name, ST_AsWKB(v.geometry) as geometry,
                COALESCE(s.green_area_m2, 0) as green_area_m2,
                COALESCE(s.ndvi_mean, 0) as ndvi_mean,
                v.Shape_Area,
                s.area_0, s.area_1, s.area_2, s.area_3, s.area_4, s.area_5, s.area_6, s.area_7,
                s.area_8, s.area_9, s.area_10, s.area_11, s.area_12, s.area_13, s.area_14, s.area_15,
                s.area_16, s.area_17, s.area_18
            FROM '{BASE_URL}/data/boundaries/Stadtviertel.parquet' v 
            LEFT JOIN '{BASE_URL}/data/stats/extended_stats.parquet' s ON v.name = s.name
        """
        df = con.execute(query).fetchdf()
        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # CRS Check (remote data may be in EPSG:25832)
        if not gdf.empty and gdf.total_bounds[0] > 180:
            gdf.crs = "EPSG:25832"
            gdf = gdf.to_crs("EPSG:4326")
        
        # Calculate green_pct
        if 'green_area_m2' in gdf.columns and 'Shape_Area' in gdf.columns:
            gdf['green_pct'] = (gdf['green_area_m2'] / gdf['Shape_Area']) * 100
        else:
            gdf['green_pct'] = 0.0
        
        return gdf
    except Exception as e:
        st.error(f"Error loading quarters: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_boroughs():
    """Load borough boundaries - Cloud version of utils.load_boroughs"""
    try:
        query = f"SELECT STB_NAME as name, ST_AsWKB(geometry) as geometry FROM '{BASE_URL}/data/boundaries/Stadtbezirke.parquet'"
        df = con.execute(query).fetchdf()
        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        if not gdf.empty and gdf.total_bounds[0] > 180:
            gdf.crs = "EPSG:25832"
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
    except Exception as e:
        st.error(f"Error loading boroughs: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def get_tile_to_veedel_mapping():
    """Get tile-to-Veedel mapping - Cloud version of utils.get_tile_to_veedel_mapping"""
    try:
        tiles_df = pd.read_csv(f"{BASE_URL}/data/metadata/cologne_tiles.csv", storage_options=STORAGE_OPTS)
        geometries = [box(r['Koordinatenursprung_East'], r['Koordinatenursprung_North'], 
                         r['Koordinatenursprung_East']+1000, r['Koordinatenursprung_North']+1000) for _, r in tiles_df.iterrows()]
        tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
        
        q_gdf = gpd.read_parquet(f"{BASE_URL}/data/boundaries/Stadtviertel.parquet", storage_options=STORAGE_OPTS)
        if q_gdf.crs != "EPSG:25832": q_gdf = q_gdf.to_crs("EPSG:25832")
        
        joined = gpd.sjoin(tiles_gdf, q_gdf, how="inner", predicate="intersects")
        return joined.groupby('name')['Kachelname'].apply(list).to_dict()
    except Exception as e:
        return {}

@st.cache_data(ttl=3600)
def list_available_tiles():
    """List available tiles from cloud (processed, web_optimized, or raw)"""
    try:
        fs = HfFileSystem(token=HF_TOKEN)
        tiles = set()
        
        # Check processed masks
        processed_files = fs.glob(f"datasets/{DATASET_ID}/data/processed/*_mask.tif")
        for f in processed_files:
            tiles.add(Path(f).stem.replace("_mask", ""))
        
        # Check web_optimized masks
        web_opt_files = fs.glob(f"datasets/{DATASET_ID}/data/web_optimized/*_mask.tif")
        for f in web_opt_files:
            tiles.add(Path(f).stem.replace("_mask", ""))
        
        # Also include raw tiles (for satellite view)
        raw_files = fs.glob(f"datasets/{DATASET_ID}/data/raw/*.jp2")
        for f in raw_files:
            tiles.add(Path(f).stem)
        
        return list(tiles)
    except:
        return []

def get_mosaic_data_remote(tile_names, layer_type):
    """Load and mosaic tiles - Cloud version of utils.get_mosaic_data_local"""
    fs = HfFileSystem(token=HF_TOKEN)
    sources = []
    memfiles = []
    
    try:
        for tile in tile_names:
            suffix = "_mask" if "Land Cover" in layer_type else "_ndvi" if "NDVI" in layer_type else ""
            
            paths = [
                f"datasets/{DATASET_ID}/data/web_optimized/{tile}{suffix}.tif",
                f"datasets/{DATASET_ID}/data/processed/{tile}{suffix}.tif",
            ]
            if layer_type == "Satellite":
                paths.append(f"datasets/{DATASET_ID}/data/raw/{tile}.jp2")
            
            found_bytes = None
            for p in paths:
                try:
                    with fs.open(p, "rb") as f:
                        found_bytes = f.read()
                        break
                except: continue
            
            if found_bytes:
                m = rasterio.MemoryFile(found_bytes)
                memfiles.append(m)
                sources.append(m.open())
        
        # Use shared processing logic
        result = process_mosaic(sources, layer_type)
        
        # Cleanup
        for s in sources: s.close()
        for m in memfiles: m.close()
        
        return result
    except Exception as e:
        return None, None

# 1. Page Configuration
st.set_page_config(page_title="GreenCologne (Cloud)", layout="wide")
st.title("🌿 GreenCologne (Cloud Dashboard)")

# --- Sidebar: Dataset Info ---
with st.sidebar:
    st.header("ℹ️ About This Dataset")
    
    with st.expander("📡 Data Sources", expanded=False):
        st.markdown("""
        **Satellite Imagery**  
        [OpenNRW DOP10](https://www.bezreg-koeln.nrw.de/geobasis-nrw/produkte-und-dienste/luftbild-und-satellitenbildinformationen/aktuelle-luftbild-und-0) – 10cm resolution aerial photos (2022-2025)
        
        **Administrative Boundaries**  
        [Offene Daten Köln](https://www.offenedaten-koeln.de/) – Stadtviertel & Stadtbezirke
        
        **Coverage**  
        840 tiles covering Cologne's 86 Veedels
        """)
    
    with st.expander("🤖 Model & Methodology", expanded=False):
        st.markdown("""
        **Land Cover Classification**  
        [FLAIR-Hub](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_IR_swinbase-upernet) – Deep learning semantic segmentation trained on French aerial imagery, adapted for German urban landscapes.
        
        **19 Land Cover Classes**  
        Buildings, deciduous trees, herbaceous vegetation, water, impervious surfaces, agricultural land, and more.
        
        **NDVI Calculation**  
        Normalized Difference Vegetation Index computed from NIR and Red bands:  
        `NDVI = (NIR - Red) / (NIR + Red)`
        
        **Green Area Detection**  
        Classes 4 (Deciduous), 5 (Coniferous), 17 (Herbaceous), and 18 (Agricultural) are classified as green areas.
        """)
    
    with st.expander("🙏 Acknowledgments", expanded=False):
        st.markdown("""
        - [CorrelAid](https://correlaid.org/) – Data-for-good community
        - [OpenNRW](https://www.opengeodata.nrw.de/) – Open geospatial data
        - [IGNF/FLAIR-Hub](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_IR_swinbase-upernet) – Segmentation model
        - [Stadt Köln](https://www.stadt-koeln.de/) – Open administrative data
        """)

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN missing. Set in .env or Streamlit Secrets.")

# 2. Data Loading
gdf_quarters = load_quarters_with_stats()
gdf_boroughs = load_boroughs()
tile_mapping = get_tile_to_veedel_mapping()
available_tiles = list_available_tiles()

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
     elif gdf_quarters is not None and not gdf_quarters.empty:
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
            filtered_tiles = [t for t in veedel_tiles if t in available_tiles]
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
                    
                    fig_pie = px.pie(
                        class_data, 
                        names='class_name', 
                        values='area_m2', 
                        title="Land Cover Distribution", 
                        color='class_name', 
                        color_discrete_map={row['class_name']: row['color'] for _, row in class_data.iterrows()},
                        labels={'class_name': 'Land Cover', 'area_m2': 'Area'}
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate='<b>%{label}</b><br>Area: %{value:,.0f} m² (%{percent})<extra></extra>'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
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
            mosaic_img, mosaic_bounds = get_mosaic_data_remote(tiles_to_display, layer_selection)
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
