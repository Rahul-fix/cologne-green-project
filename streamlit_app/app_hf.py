import streamlit as st
import plotly.express as px
from pathlib import Path
import duckdb
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import os
import shapely.wkb

st.set_page_config(page_title="GreenCologne (Cloud)", layout="wide")

st.title("🌿 GreenCologne (Cloud Dashboard)")

# --- Configuration ---
# --- Configuration ---
# Get secrets from Streamlit secrets (HF Spaces) or Environment Variables
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
DATASET_ID = st.secrets.get("DATASET_ID", os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")) # Default to uploaded dataset

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN not found. If the dataset is private, you must set it in Secrets or .env.")

# Paths (HTTPFS style for DuckDB with HF)
# Use the 'resolve/main' URL structure for raw file access
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"

STATS_FILE = f"{BASE_URL}/data/stats/stats.parquet"
DISTRICTS_FILE = f"{BASE_URL}/data/boundaries/Stadtviertel.parquet"
BOROUGHS_FILE = f"{BASE_URL}/data/boundaries/Stadtbezirke.parquet"
PROCESSED_PREFIX = f"{BASE_URL}/data/processed"

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    # Configure HTTP access (Header for private datasets)
    if HF_TOKEN:
        con.execute(f"SET s3_region='us-east-1';") # Dummy region often needed for S3 compat if used, but for HTTP:
        # For simple HTTPFS with DuckDB, we might need to pass headers if private.
        # DuckDB's HTTPFS supports headers.
        con.execute(f"SET http_keep_alive=false;") # Sometimes helps
        # Setting the header for all requests to huggingface.co
        # Note: DuckDB < 0.10 might differ, but generally:
        con.execute(f"SET http_headers={{'Authorization': 'Bearer {HF_TOKEN}'}};")
        
    return con

try:
    con = get_db_connection()
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# --- Data Loading with SQL ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_district_stats():
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
        
        def safe_load_wkb(x):
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None

        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        gdf['green_pct'] = (gdf['green_area_m2'] / gdf['Shape_Area']) * 100
        return gdf
    except Exception as e:
        st.error(f"Error loading stats: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_city_outline():
    try:
        query = f"""
            SELECT ST_AsWKB(ST_Union_Agg(geometry)) as geometry
            FROM '{BOROUGHS_FILE}'
        """
        df = con.execute(query).fetchdf()
        
        def safe_load_wkb(x):
            try:
                return shapely.wkb.loads(bytes(x))
            except Exception:
                return None
                
        df['geometry'] = df['geometry'].apply(safe_load_wkb)
        df = df.dropna(subset=['geometry'])
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Buffer fix (reprojecting in memory)
        gdf = gdf.to_crs("EPSG:25832")
        gdf['geometry'] = gdf['geometry'].buffer(10).buffer(-10)
        gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
    except Exception as e:
        st.error(f"Error loading outline: {e}")
        return gpd.GeoDataFrame()

# Load Data
with st.spinner("Loading data from cloud..."):
    gdf_districts = load_district_stats()
    gdf_city = load_city_outline()

if gdf_districts.empty:
    st.warning("Could not load data. Check your bucket name and permissions.")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Statistics", "🗺️ Map Visualization"])

with tab1:
    st.subheader("Green Area by Veedel")
    
    # Metrics
    total_green_m2 = gdf_districts['green_area_m2'].sum()
    total_green_ha = total_green_m2 / 10000
    st.metric("Total Green Area", f"{total_green_ha:.2f} ha")

    # Chart
    top_10 = gdf_districts.head(10)
    fig = px.bar(
        top_10, 
        x='name', 
        y='green_area_m2',
        title="Top 10 Greenest Districts (m²)",
        labels={'green_area_m2': 'Green Area (m²)', 'name': 'District'},
        color='green_area_m2',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig)

with tab2:
    st.subheader("Satellite Imagery Analysis")
    st.info("Note: Raster overlays (NDVI/Mask) require direct file access or signed URLs. Currently showing vector analysis only for speed.")
    
    # Create Map
    # Center on Cologne
    m = folium.Map(location=[50.9375, 6.9603], zoom_start=11, tiles="CartoDB positron")
    
    # Base Maps
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
    folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Satellite (Google)", overlay=False, control=True, show=False).add_to(m)

    # District Analysis
    folium.Choropleth(
        geo_data=gdf_districts,
        name="District Analysis (Green %)",
        data=gdf_districts,
        columns=['name', 'green_pct'],
        key_on='feature.properties.name',
        fill_color='RdYlGn',
        fill_opacity=0.6,
        line_opacity=0.2,
        legend_name='Green Space Percentage (%)',
        highlight=True
    ).add_to(m)
    
    folium.GeoJson(
        gdf_districts,
        name="District Tooltips",
        style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'green_pct', 'green_area_m2'],
            aliases=['District:', 'Green %:', 'Green Area (m²):'],
            localize=True,
            sticky=False
        )
    ).add_to(m)

    # City Outline
    if not gdf_city.empty:
        folium.GeoJson(
            gdf_city,
            name="Cologne City Outline",
            style_function=lambda x: {
                'fillColor': '#3388ff',
                'fillOpacity': 0.1,
                'color': '#222222',
                'weight': 2,
                'opacity': 0.8
            }
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=800, height=600)
