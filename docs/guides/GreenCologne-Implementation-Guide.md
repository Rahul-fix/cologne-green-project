# GreenCologne Implementation Guide

This guide provides step-by-step instructions to set up, process, and visualize the GreenCologne project from scratch.

## 1. Prerequisites

Ensure you have the following installed:
- **Python 3.10+**
- **uv** (Fast Python package installer):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## 2. Installation

1.  **Clone the repository** (if not already done):
    ```bash
    git clone <repo-url>
    cd final-cologne-green-project
    ```

2.  **Install Dependencies**:
    We use `uv` to manage dependencies. This command will create a virtual environment and install all required packages (including the local `flair-hub` module).
    ```bash
    uv sync
    ```

## 3. Data Preparation

We need to download satellite imagery and administrative boundaries for Cologne.

1.  **Download Sample Satellite Image**:
    Downloads a Sentinel-2/DOP tile for the Cologne Cathedral area.
    ```bash
    uv run scripts/00_download_sample.py
    ```

2.  **Download Administrative Boundaries**:
    Downloads `Stadtviertel` (Districts) and `Stadtbezirk` (Boroughs) from Open Data Cologne.
    ```bash
    uv run scripts/03_download_boundaries.py
    ```

3.  **Convert Boundaries to Parquet**:
    Converts shapefiles to GeoParquet for efficient processing.
    ```bash
    uv run scripts/04_convert_boundaries.py
    ```

## 4. Processing Pipeline

We use the FLAIR-HUB model to detect vegetation.

1.  **Run Inference & Processing**:
    - Calculates NDVI.
    - Runs the FLAIR-HUB pre-trained model (`flairhub_zonal`) to generate a segmentation mask.
    - Outputs processed TIFFs to `data/processed/`.
    ```bash
    uv run scripts/02_process_local.py
    ```

2.  **Generate Statistics**:
    - Spatially joins the vegetation mask with district boundaries.
    - Calculates the green area (m²) for each district.
    - Saves results to `data/stats/stats.parquet`.
    ```bash
    uv run scripts/05_generate_stats.py
    ```

## 5. Running the Dashboard

You can run the interactive Streamlit dashboard to visualize the results.

### Option A: Standard Local App
Uses Pandas/GeoPandas for data loading.
```bash
uv run streamlit run streamlit_app/app_local.py
```

### Option B: DuckDB-Powered App (Cloud Simulation)
Uses DuckDB with spatial extensions for SQL-based geospatial queries.
```bash
uv run streamlit run streamlit_app/app_duckdb.py
```

## Project Structure

- **`data/`**: Stores raw and processed data.
    - `processed/`: NDVI and Segmentation masks.
    - `stats/`: Aggregated statistics.
- **`scripts/`**: Python scripts for the pipeline.
    - `00_download_sample.py`: Downloads imagery.
    - `01_verify_local.py`: Verifies data integrity.
    - `02_process_local.py`: Runs inference and NDVI.
    - `03_download_boundaries.py`: Downloads vector data.
    - `04_convert_boundaries.py`: Converts vectors to Parquet.
    - `05_generate_stats.py`: Aggregates stats by district.
- **`streamlit_app/`**: Dashboard code.
    - `app_local.py`: Standard app.
    - `app_duckdb.py`: DuckDB backend app.
- **`DL_cologne_green/`**: FLAIR-HUB model and configuration.

## Troubleshooting

- **Map not visible?** Ensure you have an internet connection for the base maps (CartoDB/OSM).
- **Missing files?** Run the scripts in order (00 -> 03 -> 04 -> 02 -> 05).
- **DuckDB Error?** If `app_duckdb.py` fails, ensure the `spatial` extension is compatible with your system (usually handled automatically).
