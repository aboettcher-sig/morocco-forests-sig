# Morocco Forests SIG: AI Change Detection for Illegal Logging 🌲🛰️

This repository contains the data engineering and fetching pipeline for the Morocco Department of Forestry's AI-based change detection system. It automates the retrieval of high-resolution satellite imagery (Planet), multispectral time-series (Sentinel, Landsat), and climate/terrain data (Google Earth Engine) over specific Areas of Interest (AOIs), and ingests them into a centralized PostGIS database for machine learning analysis.

---

##  Architecture & Workflow

The pipeline is designed around a central PostgreSQL/PostGIS database. The general workflow is:
1. **Spin up the database** using Docker.
2. **Load your AOI** (Area of Interest) polygons into the database.
3. **Fetch satellite data** (GEE and Planet) using the AOIs stored in the database.
4. **Ingest the results** back into the database for analysis.

---

## Getting Started

### 1. Prerequisites
Ensure you have the following installed on your machine:
* **Docker & Docker Compose** (for the PostGIS database)
* **Python 3.9+**

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory (this is git-ignored for security) and add your database credentials and API keys:
```env
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=geospatial_db
POSTGRES_PORT=5432
PLANET_API_KEY=your_planet_api_key_here
```

Also, ensure you have your Earth Engine credentials set up (`credentials.json` / `token.json`) or are authenticated locally via `earthengine authenticate`.

### 3. Start the Database
Spin up the PostGIS database in the background:
```bash
docker-compose up -d
```
*(To stop the database later, run `docker-compose down`)*

---

## Usage & Commands

### Step 1: Load an AOI into the Database
Before fetching any satellite data, you must load a GeoPackage (`.gpkg`) defining your Area of Interest into the PostGIS database.
```bash
python src/database/load_aoi.py AOI/test_site_one.gpkg
```
*This extracts the geometry, ensures it is in EPSG:4326, and saves it to the `aoi_boundaries` SQL table under the name `test_site_one`.*

### Step 2: Fetch Google Earth Engine Data
Extract spectral time-series (Landsat, Sentinel) and climate/terrain data over your AOI. The script buffers the AOI, samples the pixels, and exports the data as CSVs to a folder in your Google Drive via GEE batch tasks.
```bash
python src/data_fetch/pull_gee.py --aoi test_site_one --start-year 2019 --end-year 2023 --folder GEE_Morocco_Forests
```

### Step 3: Fetch Planet Imagery
Order and download weekly high-resolution Planet basemaps clipped perfectly to your AOI. The script submits an order to Planet's Compute API and downloads the finalized `.tif` files locally.
```bash
python src/data_fetch/pull_planet.py --aoi test_site_one --out-dir data/planet_imagery --start 2023-01-01 --end 2023-12-31
```

### Step 4: Ingest GEE CSVs back into the Database
Once Earth Engine finishes exporting the CSVs to your Google Drive, download that folder locally and ingest it into PostGIS for analysis. The script automatically categorizes the files into the correct tables (e.g., `gee_landsat_data`, `gee_climate_data`).
```bash
python src/database/load_gee_csv.py --csv-dir path/to/downloaded/csvs
```

*(Note: There is also an experimental script, `pull_gee_direct.py`, which attempts to stream data directly into the database without Google Drive. However, this is only recommended for very small areas or low-resolution datasets to avoid Out-Of-Memory errors).*

---

##  Repository Structure

* **`src/`**: The core production scripts.
  * **`data_fetch/`**
    * `pull_gee.py`: Submits batch export tasks to Google Earth Engine (to Google Drive).
    * `pull_planet.py`: Submits clipping orders to Planet API and downloads `.tif` files.
    * `pull_gee_direct.py`: Experimental script to pull GEE data directly into Python RAM via HTTP (use with caution on large datasets).
  * **`database/`**
    * `db_utils.py`: Helper functions for connecting to the PostgreSQL database.
    * `load_aoi.py`: CLI tool to upload `.gpkg` boundary files to PostGIS.
    * `load_gee_csv.py`: CLI tool to ingest downloaded GEE CSVs into PostGIS tables.
    * `load_drive_to_db.py`: Utilities for pulling files directly from the Google Drive API.
* **`scripts/`**: Development and analysis notebooks.
  * **`utility/`**: Interactive Jupyter Notebooks used for prototyping the Earth Engine and Planet API logic step-by-step before they were productionized into the `src/` folder.
  * **`analysis/`**: Notebooks for exploring and modeling the data.
* **`docker-compose.yml`**: Infrastructure configuration to instantly spin up the PostGIS database.
* **`.env`**: (Git-ignored) Stores your sensitive passwords and API keys.
* **`.gitignore`**: Ensures API keys, raw data folders, and environments are not uploaded.
