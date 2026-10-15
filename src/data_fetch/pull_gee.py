import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing.pool import ThreadPool
from functools import partial
import pandas as pd
import geopandas as gpd
import requests
import ee

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Database and Drive utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine
from load_gee_csv import ingest_df_to_db, is_file_ingested, load_csv_to_db, get_table_name
from drive_utils import upload_file_to_drive

_COLLECTION_PIXEL_SIZE_CACHE = {}

DATASET_REGISTRY = {
    # Dense datasets processed via direct chunking and streamed directly into PostGIS
    "Sentinel2":    {"id": "COPERNICUS/S2_SR_HARMONIZED", "chunked": True},
    "Sentinel1":    {"id": "COPERNICUS/S1_GRD",            "chunked": True},
    # Datasets processed via GEE Drive export tasks
    "Landsat9":     {"id": "LANDSAT/LC09/C02/T1_L2",       "chunked": False},
    "Landsat8":     {"id": "LANDSAT/LC08/C02/T1_L2",       "chunked": False},
    "Landsat7":     {"id": "LANDSAT/LE07/C02/T1_L2",       "chunked": False},
    "Landsat5":     {"id": "LANDSAT/LT05/C02/T1_L2",       "chunked": False},
    "TerraClimate": {"id": "IDAHO_EPSCOR/TERRACLIMATE",     "chunked": False},
    "CFSv2":        {"id": "NOAA/CFSV2/FOR6H_HARMONIZED",  "chunked": False},
    "DynamicWorld": {"id": "GOOGLE/DYNAMICWORLD/V1",        "chunked": False},
}


def authenticate_and_initialize(project="morocco-data-509823"):
    """Authenticates and initializes Earth Engine with GCP ADC or user OAuth."""
    print(f"Initializing Google Earth Engine with project '{project}'...")
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=[
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/earthengine'
        ])
        ee.Initialize(credentials=creds, project=project)
    except Exception:
        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project)
    print("[OK] Earth Engine Initialized Successfully!")


def fetch_aoi_from_db(aoi_name: str) -> gpd.GeoDataFrame:
    """Connects to PostGIS, fetches the AOI by name, and returns the GeoDataFrame."""
    engine = get_db_engine()
    query = "SELECT name, geometry FROM aoi_boundaries WHERE name = %(name)s"
    print(f"Fetching AOI '{aoi_name}' from PostGIS...")
    gdf = gpd.read_postgis(query, con=engine, geom_col='geometry', params={"name": aoi_name})
    if gdf.empty:
        raise ValueError(f"AOI '{aoi_name}' not found in the database. Did you load it first?")
    return gdf


def get_collection_pixel_sizes(dataset_id: str):
    """
    Returns the minimum and maximum native band pixel sizes (in meters) for a dataset.
    Filters out unprojected coordinate/metadata bands (>1000m) to avoid bloated buffers.
    """
    if dataset_id not in _COLLECTION_PIXEL_SIZE_CACHE:
        asset_type = ee.data.getAsset(dataset_id)["type"]
        if asset_type == "IMAGE_COLLECTION":
            coll = ee.ImageCollection(dataset_id)
            if "S1_GRD" in dataset_id:
                coll = coll.filter(ee.Filter.eq('instrumentMode', 'IW'))
            image = ee.Image(coll.first())
        elif asset_type == "IMAGE":
            image = ee.Image(dataset_id)
        else:
            raise ValueError(f"{dataset_id} is a {asset_type}; expected IMAGE or IMAGE_COLLECTION")

        band_names = image.bandNames().getInfo()
        if not band_names:
            raise ValueError(f"No bands found in collection: {dataset_id}")

        band_scales = [
            float(image.select(band).projection().nominalScale().getInfo())
            for band in band_names
        ]
        valid_scales = [s for s in band_scales if s <= 1000] or band_scales
        _COLLECTION_PIXEL_SIZE_CACHE[dataset_id] = (min(valid_scales), max(valid_scales))
        print(f"{dataset_id}: native band scales {min(valid_scales):g}-{max(valid_scales):g} m")

    return _COLLECTION_PIXEL_SIZE_CACHE[dataset_id]


def build_aoi(gdf: gpd.GeoDataFrame, dataset_id: str = None, pixel_size: float = None):
    """Buffer gdf by 3x native pixel size, reproject to WGS84, and return an ee.FeatureCollection."""
    if pixel_size is None:
        if dataset_id is None:
            raise ValueError("Pass either dataset_id or pixel_size.")
        _, pixel_size = get_collection_pixel_sizes(dataset_id)

    buffered = gdf.copy()
    utm_crs = gdf.estimate_utm_crs()
    buffered['geometry'] = buffered.to_crs(utm_crs).buffer(pixel_size * 3).to_crs('EPSG:4326')
    return ee.FeatureCollection([ee.Feature(ee.Geometry(g.__geo_interface__)) for g in buffered['geometry']])


def request_single_export(chunk_info, folder: str, filename_prefix: str):
    """Downloads an individual chunk of sampled features via direct GEE URL."""
    chunk_collection, chunk_id = chunk_info
    url = chunk_collection.getDownloadURL(filetype='CSV')
    chunk_path = Path(folder) / f"{filename_prefix}_chunk_{chunk_id}.csv"

    for attempt in range(5):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200:
                with open(chunk_path, "w", encoding="utf-8") as out:
                    out.write(r.text)
                return
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def assemble_chunked_exports(folder: str, filename_prefix: str, engine=None, upload_to_drive: bool = True):
    """Combines chunk CSVs into one unified CSV, loads to PostGIS, and uploads to Drive."""
    chunks = sorted(list(Path(folder).glob(f"{filename_prefix}_chunk_*.csv")))
    if not chunks:
        return None

    dfs = []
    for c in chunks:
        try:
            dfs.append(pd.read_csv(c))
        except Exception as e:
            print(f"  Error reading {c.name}: {e}")

    if not dfs:
        return None

    out_df = pd.concat(dfs, ignore_index=True)
    out_df.columns = [str(col).replace(':', '_').replace('.', '_') for col in out_df.columns]
    final_path = Path(folder) / f"{filename_prefix}.csv"
    out_df.to_csv(final_path, index=False)
    print(f"  [OK] Saved {final_path.name} ({len(out_df)} rows)")

    # Cleanup temporary chunk files
    for c in chunks:
        try:
            os.remove(c)
        except Exception:
            pass

    # Direct Ingestion to PostgreSQL if engine is provided
    if engine is not None:
        ingest_df_to_db(out_df, final_path.name, engine)

    # Upload assembled CSV to Google Drive
    if upload_to_drive:
        upload_file_to_drive(final_path, drive_folder_name=folder)

    return final_path


def export_gee_dataset_chunked(dataset_id: str, aoi, start_date: str, end_date: str,
                               filename_prefix: str, folder: str = 'GEE_Exports',
                               n_helpers: int = 6, n_pixels_per_chunk: int = 3000,
                               engine=None, upload_to_drive: bool = True):
    """
    Downloads dense datasets (Sentinel-1, Sentinel-2) inside an AOI in parallel chunks.
    Bypasses GEE's 10MB text serialization constraint and streams directly into PostGIS.
    """
    os.makedirs(folder, exist_ok=True)
    final_csv = Path(folder) / f"{filename_prefix}.csv"
    if final_csv.exists() and final_csv.stat().st_size > 100:
        print(f"  [Skip] {final_csv.name} already exists on disk.")
        if engine is not None:
            if not is_file_ingested(final_csv.name, engine):
                try:
                    df = pd.read_csv(final_csv)
                    ingest_df_to_db(df, final_csv.name, engine)
                except Exception as e:
                    print(f"  [PostgreSQL] Note on {final_csv.name}: {e}")
            else:
                print(f"  [Skip] {final_csv.name} -> {get_table_name(final_csv.name)} (already ingested)")
        if upload_to_drive:
            upload_file_to_drive(final_csv, drive_folder_name=folder)
        return final_csv

    try:
        collection = ee.ImageCollection(dataset_id).filterBounds(aoi).filterDate(ee.Date(start_date), ee.Date(end_date))
        if "S1_GRD" in dataset_id:
            collection = (
                collection
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .select(['VV', 'VH', 'angle'])
            )

        if collection.size().getInfo() == 0:
            print(f"Skipping {filename_prefix}: No images found.")
            return None

        min_pixel_size, _ = get_collection_pixel_sizes(dataset_id)
        sampled_fc = collection.map(
            lambda img: img.sample(region=aoi.geometry(), scale=min_pixel_size, geometries=True, dropNulls=False)
        ).flatten()
        fc_size = sampled_fc.size().getInfo()

        if fc_size == 0:
            print(f"Skipping {filename_prefix}: 0 pixels sampled.")
            return None

        n_chunks = max(int(fc_size // n_pixels_per_chunk) + (1 if fc_size % n_pixels_per_chunk else 0), 1)
        print(f"Pulling {filename_prefix}: {fc_size} pixels across {n_chunks} parallel chunks...")

        chunked_collections = [
            (ee.FeatureCollection(sampled_fc.toList(
                count=min(n_pixels_per_chunk, fc_size - i * n_pixels_per_chunk),
                offset=i * n_pixels_per_chunk
            )), i)
            for i in range(n_chunks)
        ]

        pool = ThreadPool(processes=min(n_helpers, n_chunks))
        pool.map(partial(request_single_export, folder=folder, filename_prefix=filename_prefix), chunked_collections)
        pool.close()
        pool.join()

        return assemble_chunked_exports(folder, filename_prefix, engine=engine, upload_to_drive=upload_to_drive)

    except Exception as e:
        print(f"Failed chunked export for {dataset_id} ({filename_prefix}): {e}")
        return None


def export_gee_dataset(dataset_id: str, aoi, start_date: str, end_date: str,
                       filename_prefix: str, folder: str = 'GEE_Exports'):
    """Submits a standard GEE background export task to Google Drive."""
    try:
        collection = ee.ImageCollection(dataset_id).filterBounds(aoi).filterDate(ee.Date(start_date), ee.Date(end_date))
        if collection.size().getInfo() == 0:
            print(f"Skipping {filename_prefix}: No images found.")
            return

        min_pixel_size, _ = get_collection_pixel_sizes(dataset_id)
        sampled_fc = collection.map(
            lambda img: img.sample(region=aoi.geometry(), scale=min_pixel_size, geometries=True, dropNulls=False)
        ).flatten()

        ee.batch.Export.table.toDrive(
            collection=sampled_fc,
            description=filename_prefix,
            folder=folder,
            fileNamePrefix=filename_prefix,
            fileFormat="CSV"
        ).start()
        print(f"Started export task: {filename_prefix} (CSV)")

    except Exception as e:
        print(f"Failed to process dataset {dataset_id}: {e}")


def get_date_intervals(dataset_id: str, year: int, s2_interval: str = "monthly", months=None):
    """Returns list of (start_date, end_date, suffix) for a given year."""
    is_dense = any(ident in dataset_id.upper() for ident in ("S2", "S1", "COPERNICUS", "DYNAMICWORLD"))

    if is_dense and s2_interval == "quarterly":
        return [
            (f"{year}-01-01", f"{year}-04-01", f"{year}_Q1"),
            (f"{year}-04-01", f"{year}-07-01", f"{year}_Q2"),
            (f"{year}-07-01", f"{year}-10-01", f"{year}_Q3"),
            (f"{year}-10-01", f"{year + 1}-01-01", f"{year}_Q4"),
        ]
    elif is_dense and s2_interval == "monthly":
        selected_months = months if months else list(range(1, 13))
        return [
            (f"{year}-{m:02d}-01", f"{year}-{m+1:02d}-01" if m < 12 else f"{year+1}-01-01", f"{year}_{m:02d}")
            for m in selected_months
        ]
    return [(f"{year}-01-01", f"{year + 1}-01-01", f"{year}")]


def export_terrain_data(gdf: gpd.GeoDataFrame, aoi_name: str, drive_folder: str):
    """Submits static terrain exports (SRTM mTPI, MERIT elevation, Geomorpho90m slope & aspect)."""
    print("\n--- Submitting Static Terrain Data ---")
    try:
        aoi_ee_terrain = build_aoi(gdf, "CSP/ERGo/1_0/Global/SRTM_mTPI")
        img = ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI")
        min_p, _ = get_collection_pixel_sizes("CSP/ERGo/1_0/Global/SRTM_mTPI")
        sampled = img.sample(region=aoi_ee_terrain.geometry(), scale=min_p, geometries=True, dropNulls=False)
        ee.batch.Export.table.toDrive(
            collection=sampled, description=f"SRTM_mTPI_{aoi_name}", folder=drive_folder,
            fileNamePrefix=f"SRTM_mTPI_{aoi_name}", fileFormat="CSV"
        ).start()
        print(f"Started export task: SRTM_mTPI_{aoi_name} (CSV)")
    except Exception as e:
        print(f"Failed to submit SRTM mTPI: {e}")

    try:
        aoi_ee_90 = build_aoi(gdf, pixel_size=90)
        layers = {
            "elevation": ee.Image("MERIT/DEM/v1_0_3").select("dem"),
            "slope": ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/slope").filterBounds(aoi_ee_90).mosaic(),
            "aspect": ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/aspect").filterBounds(aoi_ee_90).mosaic(),
        }
        for key, layer_img in layers.items():
            export_filename = f"Terrain_{aoi_name}_{key}"
            sampled = layer_img.rename(key).sample(region=aoi_ee_90.geometry(), scale=90, geometries=True, dropNulls=False)
            ee.batch.Export.table.toDrive(
                collection=sampled, description=export_filename, folder=drive_folder,
                fileNamePrefix=export_filename, fileFormat="CSV"
            ).start()
            print(f"Started export task: {export_filename} (CSV)")
    except Exception as e:
        print(f"Failed to submit 90m terrain layers: {e}")


def main(aoi_name: str, drive_folder: str, start_year: int, end_year: int,
         s2_interval: str = "monthly", project: str = "morocco-data-509823",
         upload_to_drive: bool = True, datasets=None, months=None):
    authenticate_and_initialize(project=project)
    engine = get_db_engine()
    gdf = fetch_aoi_from_db(aoi_name)

    # 1. Process Time-Series Datasets
    active_datasets = DATASET_REGISTRY
    if datasets:
        selected_keys = [d.lower() for d in datasets]
        active_datasets = {k: v for k, v in DATASET_REGISTRY.items() if k.lower() in selected_keys}

    if active_datasets:
        print(f"\n--- Submitting Time Series Datasets ({start_year} to {end_year}) ---")
        for name, info in active_datasets.items():
            dataset_id = info["id"]
            is_chunked = info["chunked"]
            try:
                aoi_ee = build_aoi(gdf, dataset_id)
                for year in range(start_year, end_year + 1):
                    intervals = get_date_intervals(dataset_id, year, s2_interval=s2_interval, months=months)
                    for start_date, end_date, suffix in intervals:
                        filename = f"{name}_{aoi_name}_{suffix}"
                        if is_chunked:
                            export_gee_dataset_chunked(
                                dataset_id=dataset_id, aoi=aoi_ee, start_date=start_date, end_date=end_date,
                                filename_prefix=filename, folder=drive_folder, engine=engine, upload_to_drive=upload_to_drive
                            )
                        else:
                            export_gee_dataset(
                                dataset_id=dataset_id, aoi=aoi_ee, start_date=start_date, end_date=end_date,
                                filename_prefix=filename, folder=drive_folder
                            )
            except Exception as e:
                print(f"Failed to process {name}: {e}")

    # 2. Process Static Terrain Data
    run_terrain = datasets is None or any(d.lower() in ("terrain", "srtm", "merit") for d in datasets)
    if run_terrain:
        export_terrain_data(gdf, aoi_name, drive_folder)

    # 3. Ingest any local CSVs from folder into PostGIS
    if Path(drive_folder).exists():
        print(f"\n--- Syncing local CSVs from '{drive_folder}' into PostgreSQL ---")
        try:
            load_csv_to_db(drive_folder)
        except Exception as e:
            print(f"Note on CSV sync: {e}")

    print("\n[OK] Pipeline completed! All requested Earth Engine data processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and export GEE data to Google Drive and PostGIS.")
    parser.add_argument("--aoi", type=str, required=True, help="Name of the AOI in PostGIS (e.g., test_site_one)")
    parser.add_argument("--folder", type=str, default="GEE_Morocco_Forests", help="Drive folder / local export folder")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year for time series data")
    parser.add_argument("--end-year", type=int, default=2023, help="End year for time series data")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to pull (e.g., Sentinel2 Sentinel1). Default: all")
    parser.add_argument("--project", type=str, default="morocco-data-509823", help="GCP project ID for Earth Engine")
    parser.add_argument("--s2-interval", type=str, default="monthly", choices=["monthly", "quarterly", "yearly"],
                        help="Date slicing interval for 10m datasets (default: monthly)")
    parser.add_argument("--months", nargs="+", type=int, default=None, help="Specific months (1-12) to pull. Default: all")
    parser.add_argument("--upload-drive", action="store_true", default=True, help="Upload CSVs to Google Drive (default: True)")
    parser.add_argument("--no-upload-drive", dest="upload_drive", action="store_false", help="Disable Drive upload")

    args = parser.parse_args()
    main(
        args.aoi,
        args.folder,
        args.start_year,
        args.end_year,
        s2_interval=args.s2_interval,
        project=args.project,
        upload_to_drive=args.upload_drive,
        datasets=args.datasets,
        months=args.months
    )
