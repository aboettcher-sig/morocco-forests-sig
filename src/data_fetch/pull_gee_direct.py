import os
import sys
import time
import argparse
import pandas as pd
import geopandas as gpd
import ee

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine

_COLLECTION_PIXEL_SIZE_CACHE = {}

def authenticate_and_initialize():
    print("Initializing Google Earth Engine...")
    try:
        ee.Initialize()
    except Exception as e:
        print(f"Earth Engine not authorized. Running authentication flow...")
        ee.Authenticate()
        ee.Initialize()
    print("✅ Earth Engine Initialized Successfully!")

def fetch_aoi_from_db(aoi_name: str) -> gpd.GeoDataFrame:
    engine = get_db_engine()
    query = "SELECT name, geometry FROM aoi_boundaries WHERE name = %(name)s"
    gdf = gpd.read_postgis(query, con=engine, geom_col='geometry', params={"name": aoi_name})
    if gdf.empty:
        raise ValueError(f"AOI '{aoi_name}' not found in the database.")
    return gdf

def get_collection_pixel_sizes(dataset_id):
    if dataset_id not in _COLLECTION_PIXEL_SIZE_CACHE:
        asset_type = ee.data.getAsset(dataset_id)["type"]
        if asset_type == "IMAGE_COLLECTION":
            image = ee.Image(ee.ImageCollection(dataset_id).first())
        else:
            image = ee.Image(dataset_id)
        band_names = image.bandNames().getInfo()
        band_scales = [float(image.select(band).projection().nominalScale().getInfo()) for band in band_names]
        _COLLECTION_PIXEL_SIZE_CACHE[dataset_id] = (min(band_scales), max(band_scales))
    return _COLLECTION_PIXEL_SIZE_CACHE[dataset_id]

def build_aoi(gdf, dataset_id=None, pixel_size=None):
    if pixel_size is None:
        _, pixel_size = get_collection_pixel_sizes(dataset_id)
    buffered = gdf.copy()
    utm_crs = gdf.estimate_utm_crs()
    buffered['geometry'] = buffered.to_crs(utm_crs).buffer(pixel_size * 3).to_crs('EPSG:4326')
    return ee.FeatureCollection([ee.Feature(ee.Geometry(geom.__geo_interface__)) for geom in buffered['geometry']])

def fetch_and_load_direct(dataset_id, aoi_ee, start_date, end_date, table_name, engine):
    """Fetches data directly into memory using getInfo() and pushes to PostGIS immediately."""
    print(f"  Fetching {dataset_id} ({start_date} to {end_date}) into RAM...")
    
    collection = ee.ImageCollection(dataset_id).filterBounds(aoi_ee).filterDate(ee.Date(start_date), ee.Date(end_date))
    count = collection.size().getInfo()
    
    if count == 0:
        print(f"    Skipping: No images found.")
        return

    min_pixel_size, _ = get_collection_pixel_sizes(dataset_id)

    def extract_pixels(image):
        return image.sample(region=aoi_ee.geometry(), scale=min_pixel_size, geometries=True, dropNulls=False)

    sampled_fc = collection.map(extract_pixels).flatten()
    
    # MAGIC HAPPENS HERE: Pull directly into python memory
    try:
        data = sampled_fc.getInfo()
        features = data.get('features', [])
        records = [f['properties'] for f in features]
        
        if not records:
            print("    No pixels fell within the boundary.")
            return
            
        df = pd.DataFrame(records)
        df['source_dataset'] = dataset_id
        df['start_date'] = start_date
        
        # Load directly to Postgres
        df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)
        print(f"    ✅ Success: Loaded {len(df)} rows into {table_name}")
        
    except Exception as e:
        print(f"    ❌ Failed to fetch/load {dataset_id}: {e}")

def main(aoi_name, start_year, end_year):
    authenticate_and_initialize()
    engine = get_db_engine()
    gdf = fetch_aoi_from_db(aoi_name)
    
    spectral_datasets = {
        "LANDSAT/LC09/C02/T1_L2": "gee_landsat_data",
        "LANDSAT/LC08/C02/T1_L2": "gee_landsat_data",
        "COPERNICUS/S2_SR_HARMONIZED": "gee_sentinel_data",
    }
    
    print(f"\n--- Initiating Direct Database Stream ({start_year} to {end_year}) ---")
    for dataset_id, table_name in spectral_datasets.items():
        try:
            aoi_ee = build_aoi(gdf, dataset_id)
            for i in range(start_year, end_year + 1):
                fetch_and_load_direct(dataset_id, aoi_ee, f"{i}-01-01", f"{i + 1}-01-01", table_name, engine)
        except Exception as e:
            print(f"Failed to initialize {dataset_id}: {e}")

    print(f"\n✅ Pipeline Complete! Data is now physically inside PostgreSQL.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream GEE data directly into PostGIS (No Google Drive needed).")
    parser.add_argument("--aoi", type=str, required=True, help="Name of the AOI in PostGIS")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    main(args.aoi, args.start_year, args.end_year)
