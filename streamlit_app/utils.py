import streamlit as st
import pandas as pd
import duckdb
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path

# --- Constants ---
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "stats" / "extended_stats.parquet"
QUARTERS_FILE = DATA_DIR / "boundaries" / "Stadtviertel.parquet"
BOROUGHS_FILE = DATA_DIR / "boundaries" / "Stadtbezirke.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_METADATA_FILE = DATA_DIR / "metadata" / "cologne_tiles.csv"

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

# --- Data Loading ---
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
        if 'green_area_m2' in gdf.columns and 'Shape_Area' in gdf.columns:
             gdf['green_pct'] = (gdf['green_area_m2'] / gdf['Shape_Area']) * 100
        else:
             gdf['green_pct'] = 0.0
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

# --- Mosaic Logic ---
def get_mosaic_data(tile_names, layer_type):
    """
    Loads tiles, Mosaics in 25832, Reprojects to 4326, Colorizes.
    This is complex IO/numpy work, kept separate from UI.
    """
    sources = []
    try:
        for tile_name in tile_names:
            # Determine Suffix
            suffix = "_mask" if ("Land Cover" in layer_type) else "_ndvi"
            if layer_type == "Satellite": suffix = ""
            
            # Paths
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
        mosaic, out_trans = merge(sources)
        for s in sources: s.close()
        
        # 2. Reproject to WGS84
        src_crs = CRS.from_epsg(25832)
        src_height, src_width = mosaic.shape[1], mosaic.shape[2]
        dst_crs = CRS.from_epsg(4326)
        
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height, 
            *rasterio.transform.array_bounds(src_height, src_width, out_trans)
        )
        
        count = mosaic.shape[0]
        if layer_type == "Satellite" and count < 3: count = 1 
        
        dst_array = np.zeros((count, dst_height, dst_width), dtype=mosaic.dtype)
        
        reproject(
            source=mosaic, destination=dst_array,
            src_transform=out_trans, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        
        # 3. Visualization Post-Processing
        final_image = None
        
        if layer_type == "Satellite":
            if dst_array.shape[0] >= 3:
                rgb = np.moveaxis(dst_array[:3], 0, -1)
                if rgb.dtype == 'uint16':
                    p2, p98 = np.percentile(rgb[rgb > 0], (2, 98))
                    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                    final_image = (rgb * 255).astype(np.uint8)
                else:
                    final_image = rgb
                alpha = np.any(final_image > 0, axis=2).astype(np.uint8) * 255
                final_image = np.dstack((final_image, alpha))

        elif layer_type == "Land Cover":
            mask_data = dst_array[0]
            rgba = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
            for cls_id, color in FLAIR_COLORS.items(): 
                rgba[mask_data == cls_id] = color
            final_image = rgba

        elif layer_type == "NDVI":
            ndvi = dst_array[0].astype('float32')
            norm = mcolors.Normalize(vmin=-0.4, vmax=1, clip=True)(ndvi)
            cmap = plt.get_cmap('RdYlGn')
            final_image_float = cmap(norm)
            final_image = (final_image_float * 255).astype(np.uint8)
            
        # Calculate Bounds
        dst_bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
        folium_bounds = [[dst_bounds[1], dst_bounds[0]], [dst_bounds[3], dst_bounds[2]]]

        return final_image, folium_bounds

    except Exception as e:
        return None, None
