# Cologne Green Project

This project analyzes green areas in Cologne using satellite imagery and GIS data. It includes NDVI calculation, vector and raster reprojection, and visualization of results with legends and colormaps.

## 🚀 Getting Started

**👉 [Click here for the Step-by-Step Workflow Guide](WORKFLOW.md)**

This guide covers the entire process from data download to final analysis.

## Project Structure
- `data/` — Contains input boundaries, satellite images, and outputs
- `Notebooks/` — Analysis notebooks
    - `cologne_analysis_workflow.ipynb` — **Phase 5: Python Analysis** (Run this after QGIS steps)
    - `cologne_green_reprojection.ipynb` — Helper for reprojection
    - `saving_layers_with_style.ipynb` — Helper for visualization
- `cologne_analysis.gpkg` — Main GeoPackage database (created during workflow)

## Example Visualizations

### NDVI for Cologne
![NDVI visualization for Cologne (optimized for GitHub)](data/outputs/ndvi_10m_github.png)

*Original high-resolution NDVI image is available as `ndvi_10m.png` (large file, may not display on GitHub).* 

### Categorized Green Analysis by Veedel
![Veedel green categorized](data/outputs/veedel_green_categorized_resized.png)

## Main Steps (Summary)
1. **Import and reproject raster/vector data** to a common CRS (EPSG:25832)
2. **Calculate NDVI** and other green indices
3. **Clip rasters** to the Cologne city boundary
4. **Visualize and export** styled layers as images with legends

## Requirements
- QGIS with Python (PyQGIS)
- Satellite and boundary data for Cologne
- Python libraries: `geopandas`, `pandas` (for Phase 5)

## Credits
- Open data from Stadt Köln and Copernicus/Sentinel