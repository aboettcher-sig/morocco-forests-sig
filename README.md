# Morocco Forests SIG: AI Change Detection for Illegal Logging

This repository contains the data engineering and automated extraction pipeline for the Morocco Department of Forestry's AI-based change detection system. It automates the retrieval of high-resolution satellite imagery (Planet), dense multispectral and radar time-series (Sentinel-2, Sentinel-1, Landsat), and climate/terrain models (Google Earth Engine) over specific Areas of Interest (AOIs), and streams them into a centralized PostGIS database and Google Drive.

---

## Workflow Overview

1. **Area of Interest (AOI) Definition:** An AOI polygon (`.gpkg`) is uploaded into PostGIS table `aoi_boundaries`.
2. **Data Extraction (`pull_gee.py`):**
   - **Sentinel-1 & Sentinel-2:** Extracted via parallel HTTP chunk streaming directly to CSV, uploaded to Google Drive, and ingested into PostGIS without hitting Earth Engine serialization limits.
   - **Landsat, Climate Models, and Terrain:** Exported via Earth Engine batch tasks directly to Google Drive.
3. **Data Ingestion (`load_drive_to_db.py` / `load_gee_csv.py`):**
   - Automatically synchronizes and streams Google Drive exports directly into categorized PostGIS tables with duplicate avoidance and dynamic schema alignment.

---

## Key Features and Engineering Fixes

- **Parallel Chunking:** Bypasses Earth Engine's 10MB text serialization constraint (`Computed value is too large`) and `User memory limit exceeded` by partitioning queries into parallel chunks over direct HTTP streams.
- **Dedicated Radar and Optical Separation:** Separate database tables for Sentinel-1 (Radar SAR) and Sentinel-2 (Multispectral Optical).
- **Dynamic Schema Migration:** Automatically detects new satellite metadata or quality bands (such as `MSK_CLASSI_CIRRUS`, `AOT`, `WVP`) and executes `ALTER TABLE ADD COLUMN` to prevent insertion failures.
- **Deduplication:** Fast SQL checks (`is_file_ingested`) skip re-processing or re-uploading files that already exist on disk, in Google Drive, or in PostGIS.
- **Native Scale Buffering Fix:** Accurately computes native band nominal scales while filtering unprojected 1-degree coordinate bands, guaranteeing accurate 180 m AOI buffers.
- **Dual Cloud and Local Authentication:** Supports both Google Cloud Application Default Credentials (ADC) for headless deployment on GCP Cloud Run / VMs, and OAuth (`credentials.json` / `token.json`) for local development.

---

## Getting Started

### 1. Prerequisites
- Docker and Docker Compose (for PostgreSQL / PostGIS)
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory (git-ignored):
```env
POSTGRES_USER=geo_user
POSTGRES_PASSWORD=geo_password
POSTGRES_DB=geo_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
PLANET_API_KEY=your_planet_api_key_here
```

Ensure Earth Engine credentials are initialized:
```bash
earthengine authenticate
```

### 3. Start the PostGIS Database
```bash
docker-compose up -d
```
*(To stop the database: `docker-compose down`)*

---

## Usage and Workflows

### Step 1: Ingest Area of Interest (AOI)
Upload a GeoPackage (`.gpkg`) boundary file into PostGIS:
```bash
python src/database/load_aoi.py AOI/test_site_one.gpkg
```
*Stores the polygon in `aoi_boundaries` reprojected to EPSG:4326.*

---

### Step 2: Extract Earth Engine Data (`pull_gee.py`)
Run the unified extraction pipeline for any time range and target AOI:

#### Full Run (All Datasets for 2023)
```bash
python src/data_fetch/pull_gee.py --aoi test_site_one --start-year 2023 --end-year 2023 --folder GEE_Morocco_Forests
```

#### Selective Dataset Extraction
```bash
# Pull only Sentinel-2 and Sentinel-1
python src/data_fetch/pull_gee.py --aoi test_site_one --datasets Sentinel2 Sentinel1 --start-year 2023 --end-year 2023

# Pull specific months (e.g., January to March)
python src/data_fetch/pull_gee.py --aoi test_site_one --datasets Sentinel2 --start-year 2023 --end-year 2023 --months 1 2 3
```

#### CLI Options
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--aoi` | `str` | Required | Name of the AOI in PostGIS (`aoi_boundaries`) |
| `--folder` | `str` | `GEE_Morocco_Forests` | Google Drive folder and local output directory |
| `--start-year` | `int` | `2019` | Starting year for time-series extraction |
| `--end-year` | `int` | `2023` | Ending year for time-series extraction |
| `--datasets` | `list` | `all` | Specific datasets (e.g. `Sentinel2`, `Sentinel1`, `Landsat8`, `Terrain`) |
| `--months` | `list` | `all` | Specific months (1-12) to extract |
| `--s2-interval` | `str` | `monthly` | Slicing interval for 10m datasets (`monthly`, `quarterly`, `yearly`) |
| `--project` | `str` | `morocco-data-509823` | Google Cloud project ID for Earth Engine |
| `--no-upload-drive`| `flag` | `False` | Disables automatic Google Drive backup |

---

### Step 3: Stream Google Drive Exports to PostGIS (`load_drive_to_db.py`)
For batch datasets exported directly to Google Drive (e.g. Landsat, Climate models, Terrain), stream them into PostGIS without manual downloading:
```bash
python src/database/load_drive_to_db.py --folder GEE_Morocco_Forests
```

---

### Step 4: Local CSV Ingestion (`load_gee_csv.py`)
Synchronize local CSV exports into PostGIS:
```bash
python src/database/load_gee_csv.py --csv-dir GEE_Morocco_Forests
```

---

### Step 5: Planet High-Resolution Basemaps (`pull_planet.py`)
Order and download weekly high-resolution Planet basemaps clipped to your AOI:
```bash
python src/data_fetch/pull_planet.py --aoi test_site_one --out-dir data/planet_imagery --start 2023-01-01 --end 2023-12-31
```

---

## Repository Structure

```
├── AOI/                          # Boundary GeoPackages (.gpkg)
├── GEE_Morocco_Forests/          # Local cache for exported CSVs (git-ignored)
├── src/
│   ├── data_fetch/
│   │   ├── pull_gee.py           # Core GEE extraction engine (chunked + batch)
│   │   ├── drive_utils.py        # Google Drive API auth, folder lookups, and upload utilities
│   │   ├── pull_planet.py        # Planet Orders and Basemap clipping API
│   │   └── pull_gee_direct.py    # Experimental in-memory stream prototype
│   └── database/
│       ├── db_utils.py           # SQLAlchemy PostGIS engine and connection helpers
│       ├── load_aoi.py           # Ingests GeoPackages into aoi_boundaries
│       ├── load_gee_csv.py       # Ingests local CSVs into categorized PostGIS tables
│       └── load_drive_to_db.py   # Streams CSVs directly from Google Drive into PostGIS
├── docker-compose.yml            # Local PostGIS container configuration
├── requirements.txt              # Production Python package dependencies
├── .env                          # Local credentials and database config (git-ignored)
└── .gitignore                    # Protects tokens, credentials, and bulky spatial exports
```

---

## Database Tables Overview

| Table Name | Description | Key Attributes / Bands |
| :--- | :--- | :--- |
| `aoi_boundaries` | Area of Interest boundary polygons | `name`, `geometry` (EPSG:4326) |
| `gee_sentinel1_data` | Synthetic Aperture Radar (SAR) 10m | `VV`, `VH`, `angle`, `system_index` |
| `gee_sentinel2_data` | Multispectral Optical 10m Harmonized | `B1`-`B12`, `AOT`, `WVP`, `SCL`, masks |
| `gee_landsat_data` | Optical 30m (Landsat 5, 7, 8, 9) | `SR_B1`-`SR_B7`, `QA_PIXEL` |
| `gee_climate_data` | Monthly and 6-Hour Climate Models | `pr`, `tmmn`, `tmmx`, `soil`, `pdsi` |
| `gee_terrain_data` | Static Topography and Geomorphometry | `elevation`, `slope`, `aspect`, `mTPI` |
| `gee_dynamicworld_data`| Near-Real-Time Land Cover Probabilities | `water`, `trees`, `grass`, `crops`, `built` |
