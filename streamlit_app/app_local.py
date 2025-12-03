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

st.set_page_config(page_title="GreenCologne (Local)", layout="wide")

st.title("🌿 GreenCologne (Local Preview)")

# Paths
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "stats" / "stats.parquet"
DISTRICTS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
BOROUGHS_FILE = DATA_DIR / "boundaries" / "Stadtbezirke.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

if not STATS_FILE.exists():
    st.error(f"Stats file not found at {STATS_FILE}. Run scripts/05_generate_stats.py first.")
    st.stop()

# --- Data Loading with DuckDB ---
@st.cache_data
def load_stats():
    con = duckdb.connect()
    query = f"SELECT * FROM '{STATS_FILE}'"
    df = con.execute(query).df()
    con.close()
    return df

df = load_stats()

@st.cache_data
def load_districts():
    if not DISTRICTS_FILE.exists():
        return None
    import geopandas as gpd
    gdf = gpd.read_parquet(DISTRICTS_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

# --- Session State Initialization ---
if 'selected_veedel' not in st.session_state:
    st.session_state['selected_veedel'] = "All"
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [50.9375, 6.9603] # Cologne Center
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 11

def update_view_for_veedel():
    """Updates map center and zoom based on selected veedel"""
    veedel = st.session_state['selected_veedel']
    if veedel == "All":
        st.session_state['map_center'] = [50.9375, 6.9603]
        st.session_state['map_zoom'] = 11
    else:
        # Calculate centroid/bounds for the veedel
        # We need the geometry. We can get it from the districts file.
        # Since we load it later, we might need to load it here or cache it.
        pass # Logic implemented inside the main flow where GDF is available

@st.cache_data
def get_tile_to_veedel_mapping():
    """
    Creates a mapping of Veedel -> List of Tiles based on spatial intersection.
    """
    if not TILES_METADATA_FILE.exists() or not DISTRICTS_FILE.exists():
        return {}

    # Load Tiles Metadata
    tiles_df = pd.read_csv(TILES_METADATA_FILE)
    
    # Create geometries for tiles (assuming 1km x 1km)
    from shapely.geometry import box
    import geopandas as gpd
    
    geometries = []
    for _, row in tiles_df.iterrows():
        e = row['Koordinatenursprung_East']
        n = row['Koordinatenursprung_North']
        # DOP10 tiles are 1km x 1km
        geometries.append(box(e, n, e + 1000, n + 1000))
    
    tiles_gdf = gpd.GeoDataFrame(tiles_df, geometry=geometries, crs="EPSG:25832")
    
    # Load Districts
    districts_gdf = gpd.read_parquet(DISTRICTS_FILE)
    if districts_gdf.crs != "EPSG:25832":
        districts_gdf = districts_gdf.to_crs("EPSG:25832")
        
    # Spatial Join
    joined = gpd.sjoin(tiles_gdf, districts_gdf, how="inner", predicate="intersects")
    
    # Create mapping: Veedel -> [Tile Names]
    mapping = joined.groupby('name')['Kachelname'].apply(list).to_dict()
    return mapping

tile_mapping = get_tile_to_veedel_mapping()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Statistics", "🗺️ Map Visualization"])

with tab1:
    st.subheader("Green Area by Veedel (Sample Data)")

    if df.empty:
        st.warning("No data available.")
    else:
        # Metrics
        total_green_m2 = df['green_area_m2'].sum()
        total_green_ha = total_green_m2 / 10000
        st.metric("Total Green Area (Sample)", f"{total_green_ha:.2f} ha")

        # Chart
        fig = px.bar(
            df.sort_values('green_area_m2', ascending=False), 
            x='name', 
            y='green_area_m2',
            title="Green Area per District (m²)",
            labels={'green_area_m2': 'Green Area (m²)', 'name': 'District'},
            color='green_area_m2',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig)

        with st.expander("View Raw Data"):
            st.dataframe(df)

with tab2:
    st.subheader("Satellite Imagery Analysis")
    
    # Find processed files (masks)
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    
    if not mask_files:
        st.warning("No processed images found in data/processed/")
    else:
        # Get all available processed tiles
        available_tiles = set(f.stem.replace("_mask", "") for f in mask_files)
        
        # --- Veedel Selection ---
        # Get all districts to show in dropdown, even if no tiles
        all_districts_gdf = load_districts()
        all_veedel_names = sorted(all_districts_gdf['name'].tolist()) if all_districts_gdf is not None else []
        
        veedel_options = ["All"] + all_veedel_names
        
        # Callback to handle dropdown change
        def on_veedel_change():
            veedel = st.session_state['selected_veedel']
            if veedel == "All":
                st.session_state['map_center'] = [50.9375, 6.9603]
                st.session_state['map_zoom'] = 11
            else:
                gdf = load_districts()
                if gdf is not None:
                    selected_geom = gdf[gdf['name'] == veedel]
                    if not selected_geom.empty:
                        # Reproject to projected CRS for accurate centroid
                        # Use EPSG:25832 (UTM 32N)
                        geom_projected = selected_geom.to_crs("EPSG:25832")
                        centroid = geom_projected.geometry.centroid.iloc[0]
                        
                        # Project centroid back to WGS84
                        from shapely.geometry import Point
                        import geopandas as gpd
                        centroid_gdf = gpd.GeoDataFrame(geometry=[centroid], crs="EPSG:25832")
                        centroid_wgs84 = centroid_gdf.to_crs("EPSG:4326").geometry.iloc[0]
                        
                        st.session_state['map_center'] = [centroid_wgs84.y, centroid_wgs84.x]
                        st.session_state['map_zoom'] = 14 # Good zoom for a district

        selected_veedel = st.selectbox(
            "Select Veedel (District)", 
            veedel_options, 
            key='selected_veedel',
            on_change=on_veedel_change
        )
        
        # Filter tiles based on Veedel
        if selected_veedel != "All":
            veedel_tiles = set(tile_mapping.get(selected_veedel, []))
            
            if not veedel_tiles:
                 st.info(f"No tiles found for {selected_veedel} in the current dataset.")
            
            # Let's expand available_tiles to include raw files if we are in on-demand mode
            raw_files = list((DATA_DIR / "raw").glob("*.jp2"))
            available_raw = set(f.stem for f in raw_files)
            all_available = available_tiles.union(available_raw)
            
            filtered_tiles = [t for t in veedel_tiles if t in all_available]
            
            if not filtered_tiles and veedel_tiles:
                 st.warning(f"Tiles exist for {selected_veedel} but are not downloaded/processed yet.")
                 tile_options = []
            elif not filtered_tiles:
                 tile_options = []
            else:
                tile_options = sorted(filtered_tiles)
        else:
            # Show all processed + raw
            raw_files = list((DATA_DIR / "raw").glob("*.jp2"))
            available_raw = set(f.stem for f in raw_files)
            tile_options = sorted(list(available_tiles.union(available_raw)))

        selected_tile = st.selectbox("Select Tile", tile_options)
        
        # Option to show all tiles in the current list (filtered by Veedel)
        show_all_tiles = st.checkbox("Show all listed tiles", value=True)
        
        tiles_to_display = []
        if show_all_tiles:
            # Safety limit to prevent browser crash
            MAX_TILES_TO_SHOW = 10
            if len(tile_options) > MAX_TILES_TO_SHOW:
                st.warning(f"Showing first {MAX_TILES_TO_SHOW} tiles out of {len(tile_options)} to avoid performance issues.")
                tiles_to_display = tile_options[:MAX_TILES_TO_SHOW]
            else:
                tiles_to_display = tile_options
        else:
            if selected_tile:
                tiles_to_display = [selected_tile]
        
        # Layer selection
        # Layer selection
        layer_type = st.radio("Select Layer", ["NDVI", "Segmentation Mask", "Raw Satellite (RGB)"], horizontal=True)
        
        # Paths
        ndvi_path = PROCESSED_DIR / f"{selected_tile}_ndvi.tif"
        mask_path = PROCESSED_DIR / f"{selected_tile}_mask.tif"
        raw_path = DATA_DIR / "raw" / f"{selected_tile}.jp2"
        
        # Get bounds from the *selected* tile to center the map (or the first one if none selected)
        # We still use selected_tile for centering logic
        bounds = None
        crs = None
        
        # Helper to get bounds for a tile
        def get_tile_bounds(tile_name):
            t_ndvi = PROCESSED_DIR / f"{tile_name}_ndvi.tif"
            t_mask = PROCESSED_DIR / f"{tile_name}_mask.tif"
            t_raw = DATA_DIR / "raw" / f"{tile_name}.jp2"
            
            ref = None
            if t_mask.exists(): ref = t_mask
            elif t_ndvi.exists(): ref = t_ndvi
            elif t_raw.exists(): ref = t_raw
            
            if ref:
                with rasterio.open(ref) as src:
                    return src.bounds, src.crs
            return None, None

        if selected_tile:
             bounds, crs = get_tile_bounds(selected_tile)
        
        if bounds and crs:
            # Reproject bounds to WGS84 for Folium
            try:
                wgs84_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
            except Exception:
                # Fallback for weird CRS definitions (like the EngineeringCRS issue)
                # Assuming it is actually EPSG:25832 (ETRS89 UTM 32N) which is standard for NRW
                try:
                    src_crs = rasterio.crs.CRS.from_epsg(25832)
                    wgs84_bounds = transform_bounds(src_crs, 'EPSG:4326', *bounds)
                except Exception as e:
                    st.error(f"Failed to reproject bounds: {e}")
                    wgs84_bounds = None

            if wgs84_bounds:
                # wgs84_bounds is (min_lon, min_lat, max_lon, max_lat)
                
                center_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
                center_lon = (wgs84_bounds[0] + wgs84_bounds[2]) / 2
                
                # Create Map with Base Layer Options
                # Use session state for center/zoom
                m = folium.Map(
                    location=st.session_state['map_center'], 
                    zoom_start=st.session_state['map_zoom'], 
                    tiles="CartoDB positron"
                )
                
                # Add other Base Maps
                # Note: Adding them as TileLayers with control=True allows switching.
                # We add them AFTER the map creation.
                
                folium.TileLayer(
                    tiles="OpenStreetMap",
                    name="OpenStreetMap",
                    control=True,
                    show=False 
                ).add_to(m)
                
                # Google Satellite (Hybrid)
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                    attr="Google",
                    name="Satellite (Google)",
                    overlay=False,
                    control=True,
                    show=False
                ).add_to(m)

                # --- District Analysis Layer ---
                # Load boundaries
                gdf = load_districts()
                if gdf is not None:
                    # Merge with stats to get green percentage
                    
                    # Merge with stats to get green percentage
                    # df has 'name' and 'green_area_m2'
                    # gdf has 'name' and 'Shape_Area'
                    
                    # Join
                    merged_gdf = gdf.merge(df, on='name', how='left')
                    
                    # Calculate Percentage
                    # Handle missing data (fill with 0)
                    merged_gdf['green_area_m2'] = merged_gdf['green_area_m2'].fillna(0)
                    
                    # Calculate percentage (Shape_Area is in m2)
                    merged_gdf['green_pct'] = (merged_gdf['green_area_m2'] / merged_gdf['Shape_Area']) * 100
                    
                    # Create Choropleth
                    # Create Interactive GeoJson Layer
                    # We use GeoJson instead of Choropleth for better click handling
                    
                    # Define style function
                    def style_function(feature):
                        # Basic style
                        return {
                            'fillColor': '#ff000000', # Transparent fill by default
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.0
                        }

                    # Highlight function
                    def highlight_function(feature):
                        return {
                            'fillColor': '#ffff00',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.3
                        }

                    # Add the layer
                    geojson_layer = folium.GeoJson(
                        merged_gdf,
                        name="Districts (Click to Select)",
                        style_function=lambda x: {
                            'fillColor': 'green' if x['properties']['green_pct'] > 0 else 'gray',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.4
                        },
                        highlight_function=lambda x: {'weight': 3, 'color': 'blue'},
                        tooltip=folium.GeoJsonTooltip(
                            fields=['name', 'green_pct', 'green_area_m2'],
                            aliases=['District:', 'Green %:', 'Green Area (m²):'],
                            localize=True
                        ),
                        popup=folium.GeoJsonPopup(fields=['name'], labels=False) # Popup helps with clicking
                    )
                    geojson_layer.add_to(m)

                    # Update map view if a specific veedel is selected
                    if selected_veedel != "All":
                        # Find the geometry
                        selected_geom = merged_gdf[merged_gdf['name'] == selected_veedel]
                        if not selected_geom.empty:
                            # Get centroid
                            centroid = selected_geom.geometry.centroid.iloc[0]
                            # Update session state if it was a manual selection (not map click which is handled later)
                            # Actually, we should just update the map object here, but st_folium doesn't support dynamic zoom updates easily without reruns.
                            # We rely on the session state being passed to folium.Map()
                            pass

                # --- City Outline Layer ---
                # Load Stadtbezirk (Boroughs) to create city outline
                # Note: File was saved as Stadtbezirke.parquet in conversion script
                boroughs_path = BOROUGHS_FILE
                if boroughs_path.exists():
                    import geopandas as gpd
                    boroughs_gdf = gpd.read_parquet(boroughs_path)
                    
                    if boroughs_gdf.crs != "EPSG:4326":
                        boroughs_gdf = boroughs_gdf.to_crs("EPSG:4326")
                    
                    # Dissolve to get the outer boundary of Cologne
                    # Add a constant column to dissolve all into one
                    boroughs_gdf['common'] = 1
                    city_boundary = boroughs_gdf.dissolve(by='common')
                    
                    # Fix topology artifacts (broken lines) by buffering
                    # Reproject to a projected CRS (UTM 32N - EPSG:25832) for correct buffering
                    city_boundary = city_boundary.to_crs("EPSG:25832")
                    city_boundary['geometry'] = city_boundary['geometry'].buffer(10).buffer(-10) # Buffer in meters
                    # Project back to WGS84 for Folium
                    city_boundary = city_boundary.to_crs("EPSG:4326")
                    
                    folium.GeoJson(
                        city_boundary,
                        name="Cologne City Outline",
                        style_function=lambda x: {
                            'fillColor': '#3388ff',   # Blueish fill
                            'fillOpacity': 0.1,       # Subtle fill
                            'color': '#222222',       # Dark grey/Black outline
                            'weight': 2,              # Thinner line (was 4)
                            'opacity': 0.8            # Outline opacity
                        }
                    ).add_to(m)
                else:
                    st.error(f"Boroughs file not found at {boroughs_path}")


                # Loop through tiles to display
                # We create a FeatureGroup for the tiles
                tile_layer_group = folium.FeatureGroup(name=f"{layer_type} (Tiles)")
                
                for tile_name in tiles_to_display:
                    t_ndvi = PROCESSED_DIR / f"{tile_name}_ndvi.tif"
                    t_mask = PROCESSED_DIR / f"{tile_name}_mask.tif"
                    t_raw = DATA_DIR / "raw" / f"{tile_name}.jp2"
                    
                    # Get bounds for this specific tile
                    t_bounds, t_crs = get_tile_bounds(tile_name)
                    if not t_bounds or not t_crs:
                        continue
                        
                    # Reproject bounds
                    try:
                        t_wgs84_bounds = transform_bounds(t_crs, 'EPSG:4326', *t_bounds)
                    except Exception:
                         try:
                            src_crs = rasterio.crs.CRS.from_epsg(25832)
                            t_wgs84_bounds = transform_bounds(src_crs, 'EPSG:4326', *t_bounds)
                         except:
                            continue
                            
                    if layer_type == "NDVI":
                        # ... (NDVI logic) ...
                        # We need to adapt the logic to load from t_raw if on-demand
                        # For simplicity, let's reuse the logic but inside the loop
                        
                        # Check for pre-calculated
                        if t_ndvi.exists():
                             with rasterio.open(t_ndvi) as src:
                                data = src.read(1)
                                # Handle Int16 scaling
                                if data.dtype == 'int16':
                                    data = data.astype('float32') * 0.0001
                                
                                # Colormap
                                norm = mcolors.Normalize(vmin=-1, vmax=1)
                                cmap = plt.get_cmap('RdYlGn')
                                rgba = cmap(norm(data))
                                
                                folium.raster_layers.ImageOverlay(
                                    image=rgba,
                                    bounds=[[t_wgs84_bounds[1], t_wgs84_bounds[0]], [t_wgs84_bounds[3], t_wgs84_bounds[2]]],
                                    opacity=0.7,
                                    name=f"NDVI - {tile_name}"
                                ).add_to(tile_layer_group)
                        elif t_raw.exists():
                             # On-demand
                             with rasterio.open(t_raw) as src:
                                # Downsample for performance
                                MAX_DIM = 800 # Slightly smaller for multi-tile
                                scale = MAX_DIM / max(src.width, src.height)
                                if scale < 1:
                                    out_shape = (src.count, int(src.height * scale), int(src.width * scale))
                                    data = src.read(out_shape=out_shape, resampling=Resampling.bilinear)
                                else:
                                    data = src.read()
                                    
                                red = data[0].astype('float32')
                                nir = data[3].astype('float32')
                                ndvi = (nir - red) / (nir + red + 1e-8)
                                
                                norm = mcolors.Normalize(vmin=-1, vmax=1)
                                cmap = plt.get_cmap('RdYlGn')
                                rgba = cmap(norm(ndvi))
                                
                                folium.raster_layers.ImageOverlay(
                                    image=rgba,
                                    bounds=[[t_wgs84_bounds[1], t_wgs84_bounds[0]], [t_wgs84_bounds[3], t_wgs84_bounds[2]]],
                                    opacity=0.7,
                                    name=f"NDVI - {tile_name}"
                                ).add_to(tile_layer_group)

                    elif layer_type == "Segmentation Mask" and t_mask.exists():
                        with rasterio.open(t_mask) as src:
                             # Downsample
                            MAX_DIM = 800
                            scale = MAX_DIM / max(src.width, src.height)
                            if scale < 1:
                                out_shape = (src.count, int(src.height * scale), int(src.width * scale))
                                data = src.read(out_shape=out_shape, resampling=Resampling.nearest)
                            else:
                                data = src.read()
                                
                            mask_data = data[0]
                            # Create RGBA
                            rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
                            # 1=Green (Tree), 2=Meadow (low veg) - Example mapping
                            # Trees: Dark Green
                            rgba[mask_data == 1] = [0, 100, 0, 255]
                            # Low Veg: Light Green
                            rgba[mask_data == 2] = [144, 238, 144, 255]
                            
                            folium.raster_layers.ImageOverlay(
                                image=rgba,
                                bounds=[[t_wgs84_bounds[1], t_wgs84_bounds[0]], [t_wgs84_bounds[3], t_wgs84_bounds[2]]],
                                opacity=0.9,
                                name=f"Mask - {tile_name}"
                            ).add_to(tile_layer_group)

                    elif layer_type == "Raw Satellite (RGB)" and t_raw.exists():
                        with rasterio.open(t_raw) as src:
                            MAX_DIM = 800
                            scale = MAX_DIM / max(src.width, src.height)
                            if scale < 1:
                                out_shape = (src.count, int(src.height * scale), int(src.width * scale))
                                data = src.read(out_shape=out_shape, resampling=Resampling.bilinear)
                            else:
                                data = src.read()
                            
                            if data.shape[0] >= 3:
                                r = data[0]; g = data[1]; b = data[2]
                                rgb = np.dstack((r, g, b))
                                if src.dtypes[0] == 'uint8':
                                    rgb_norm = rgb / 255.0
                                else:
                                    p2, p98 = np.percentile(rgb, (2, 98))
                                    rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                                    
                                folium.raster_layers.ImageOverlay(
                                    image=rgb_norm,
                                    bounds=[[t_wgs84_bounds[1], t_wgs84_bounds[0]], [t_wgs84_bounds[3], t_wgs84_bounds[2]]],
                                    opacity=1.0,
                                    name=f"RGB - {tile_name}"
                                ).add_to(tile_layer_group)
                
                tile_layer_group.add_to(m)

                folium.LayerControl().add_to(m)
                
                # Render Map and Capture Clicks
                map_output = st_folium(m, width=800, height=600, returned_objects=["last_object_clicked"])
                
                # Handle Clicks
                if map_output['last_object_clicked']:
                    clicked_props = map_output['last_object_clicked'].get('properties')
                    if clicked_props and 'name' in clicked_props:
                        clicked_veedel = clicked_props['name']
                        if clicked_veedel != st.session_state['selected_veedel']:
                            st.session_state['selected_veedel'] = clicked_veedel
                            # Calculate new center/zoom
                            # We need to access the GDF again. 
                            # Since we are inside the loop where GDF is loaded, we can use it.
                            # But 'merged_gdf' is local to the if block above.
                            # We should probably move GDF loading outside or access it here.
                            # For now, let's just trigger a rerun, and the 'update_view_for_veedel' logic (if we implement it fully) 
                            # or the 'if selected_veedel != "All"' block will handle it.
                            
                            # To implement auto-zoom on click, we need to update map_center/zoom in session state
                            # We can reload the GDF briefly to get the centroid
                            if boundaries_path.exists():
                                import geopandas as gpd
                                gdf_temp = gpd.read_parquet(boundaries_path)
                                if gdf_temp.crs != "EPSG:4326":
                                    gdf_temp = gdf_temp.to_crs("EPSG:4326")
                                sel_geom = gdf_temp[gdf_temp['name'] == clicked_veedel]
                                if not sel_geom.empty:
                                    centroid = sel_geom.geometry.centroid.iloc[0]
                                    st.session_state['map_center'] = [centroid.y, centroid.x]
                                    st.session_state['map_zoom'] = 14 # Zoom in closer
                            
                            st.rerun()
            

            
        else:
            st.error("Could not determine bounds for map.")

st.sidebar.info("Local Preview Mode")
