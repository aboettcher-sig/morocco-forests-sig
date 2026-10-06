import os
import sys
import time
import argparse
import geopandas as gpd
import ee

# Add the parent directory to the system path so we can import db_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine

_COLLECTION_PIXEL_SIZE_CACHE = {}

def authenticate_and_initialize():
    """Authenticates and initializes the Google Earth Engine session."""
    print("Initializing Google Earth Engine...")
    try:
        # Try to initialize with default credentials (e.g. from `earthengine authenticate`)
        ee.Initialize()
    except Exception as e:
        print(f"Earth Engine not authorized ({e}). Running authentication flow...")
        ee.Authenticate()
        ee.Initialize()
    print(" Earth Engine Initialized Successfully!")

def fetch_aoi_from_db(aoi_name: str) -> gpd.GeoDataFrame:
    """Connects to PostGIS, fetches the AOI by name, and returns the GeoDataFrame."""
    engine = get_db_engine()
    query = "SELECT name, geometry FROM aoi_boundaries WHERE name = %(name)s"
    
    print(f"Fetching AOI '{aoi_name}' from PostGIS...")
    gdf = gpd.read_postgis(query, con=engine, geom_col='geometry', params={"name": aoi_name})
    
    if gdf.empty:
        raise ValueError(f"AOI '{aoi_name}' not found in the database. Did you load it first?")
    
    return gdf

def get_collection_pixel_sizes(dataset_id):
    """Return the minimum and maximum native band pixel sizes for a dataset."""
    if dataset_id not in _COLLECTION_PIXEL_SIZE_CACHE:
        asset_type = ee.data.getAsset(dataset_id)["type"]
        if asset_type == "IMAGE_COLLECTION":
            image = ee.Image(ee.ImageCollection(dataset_id).first())
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

        _COLLECTION_PIXEL_SIZE_CACHE[dataset_id] = (min(band_scales), max(band_scales))
        print(f"{dataset_id}: native band scales {min(band_scales):g}-{max(band_scales):g} m")

    return _COLLECTION_PIXEL_SIZE_CACHE[dataset_id]


def build_aoi(gdf, dataset_id=None, pixel_size=None):
    """Buffer gdf by 3x the largest native band pixel size, reproject to WGS84, and return an ee.FeatureCollection."""
    if pixel_size is None:
        if dataset_id is None:
            raise ValueError("Pass either dataset_id or pixel_size.")
        _, pixel_size = get_collection_pixel_sizes(dataset_id)

    buffered = gdf.copy()
    # Estimate UTM to do accurate buffering in meters
    utm_crs = gdf.estimate_utm_crs()
    buffered['geometry'] = (
        buffered
        .to_crs(utm_crs)
        .buffer(pixel_size * 3)
        .to_crs('EPSG:4326')
    )

    return ee.FeatureCollection(
        [ee.Feature(ee.Geometry(geom.__geo_interface__)) for geom in buffered['geometry']]
    )

def export_gee_dataset(dataset_id, aoi, start_date, end_date, filename_prefix, folder='GEE_Exports', export_format="CSV"):
    """Export images inside an AOI from a GEE dataset for a given time range."""
    try:
        collection = ee.ImageCollection(dataset_id).filterBounds(aoi).filterDate(ee.Date(start_date), ee.Date(end_date))

        if export_format == "GeoTIFF":
            image_list = collection.toList(collection.size())
            count = image_list.size().getInfo()
            if count == 0:
                print(f"Skipping {filename_prefix}: No images found.")
                return
            
            print(f"Exporting {count} images from {dataset_id}...")
            for i in range(count):
                image = ee.Image(image_list.get(i))
                image_id = image.get("system:index").getInfo()
                export_filename = f"{filename_prefix}_{image_id}"

                export_task = ee.batch.Export.image.toDrive(
                    image=image.clip(aoi),
                    description=export_filename,
                    folder=folder,
                    fileNamePrefix=export_filename,
                    fileFormat="GeoTIFF",
                    region=aoi,
                    maxPixels=1e13
                )
                export_task.start()
                print(f"Started export: {export_filename} ({export_format})")

        elif export_format == "CSV":
            min_pixel_size, _ = get_collection_pixel_sizes(dataset_id)
            
            # Check if collection is empty before running .map()
            if collection.size().getInfo() == 0:
                print(f"Skipping {filename_prefix}: No images found.")
                return

            def extract_pixels(image):
                return image.sample(region=aoi.geometry(), scale=min_pixel_size, geometries=True, dropNulls=False)

            sampled_fc = collection.map(extract_pixels).flatten()
            export_task = ee.batch.Export.table.toDrive(
                collection=sampled_fc,
                description=filename_prefix,
                folder=folder,
                fileNamePrefix=filename_prefix,
                fileFormat="CSV"
            )
            export_task.start()
            print(f"Started export: {filename_prefix} (CSV)")

    except Exception as e:
        print(f"Failed to process dataset {dataset_id}: {e}")

def export_gee_image(dataset_id, aoi, filename_prefix, folder='GEE_Exports', export_format="CSV"):
    """Export a single native GEE Image (like SRTM)."""
    try:
        image = ee.Image(dataset_id)
        if export_format == "CSV":
            min_pixel_size, _ = get_collection_pixel_sizes(dataset_id)
            sampled = image.sample(region=aoi.geometry(), scale=min_pixel_size, geometries=True, dropNulls=False)
            export_task = ee.batch.Export.table.toDrive(
                collection=sampled,
                description=filename_prefix,
                folder=folder,
                fileNamePrefix=filename_prefix,
                fileFormat="CSV"
            )
            export_task.start()
            print(f"Started export: {filename_prefix} (CSV)")
    except Exception as e:
        print(f"Failed to process image {dataset_id}: {e}")

def main(aoi_name, drive_folder, start_year, end_year):
    authenticate_and_initialize()
    
    # 1. Fetch AOI from PostGIS
    gdf = fetch_aoi_from_db(aoi_name)
    
    # 2. Spectral Time Series
    spectral_datasets = {
        "Landsat9":  "LANDSAT/LC09/C02/T1_L2",
        "Landsat8":  "LANDSAT/LC08/C02/T1_L2",
        "Landsat7":  "LANDSAT/LE07/C02/T1_L2",
        "Landsat5":  "LANDSAT/LT05/C02/T1_L2",
        "Sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
        "Sentinel1": "COPERNICUS/S1_GRD",
    }
    
    print(f"\n--- Submitting Spectral Time Series ({start_year} to {end_year}) ---")
    for name, dataset_id in spectral_datasets.items():
        try:
            aoi_ee = build_aoi(gdf, dataset_id)
            for i in range(start_year, end_year + 1):
                export_gee_dataset(
                    dataset_id=dataset_id,
                    aoi=aoi_ee,
                    start_date=f"{i}-01-01",
                    end_date=f"{i + 1}-01-01",
                    filename_prefix=f"{name}_{aoi_name}_{i}",
                    export_format="CSV",
                    folder=drive_folder,
                )
        except Exception as e:
            print(f"Failed to initialize {name}: {e}")

    # 3. Static Terrain Data (SRTM, MERIT DEM, Geomorpho90m)
    print("\n--- Submitting Static Terrain Data ---")
    aoi_ee_terrain = build_aoi(gdf, "CSP/ERGo/1_0/Global/SRTM_mTPI")
    export_gee_image("CSP/ERGo/1_0/Global/SRTM_mTPI", aoi_ee_terrain, f"SRTM_mTPI_{aoi_name}", folder=drive_folder)

    aoi_ee_90 = build_aoi(gdf, pixel_size=90)
    elevation = ee.Image("MERIT/DEM/v1_0_3").select("dem").rename("elevation")
    slope = ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/slope").filterBounds(aoi_ee_90).mosaic().rename("slope")
    aspect = ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/aspect").filterBounds(aoi_ee_90).mosaic().rename("aspect")
    
    for key, img in {"elevation": elevation, "slope": slope, "aspect": aspect}.items():
        export_filename = f"Terrain_{aoi_name}_{key}"
        sampled = img.sample(region=aoi_ee_90.geometry(), scale=90, geometries=True, dropNulls=False)
        ee.batch.Export.table.toDrive(
            collection=sampled, description=export_filename, folder=drive_folder, 
            fileNamePrefix=export_filename, fileFormat="CSV"
        ).start()
        print(f"Started export: {export_filename} (CSV)")

    # 4. Climate Models (TerraClimate, CFSv2, DynamicWorld)
    print(f"\n--- Submitting Climate Models ({start_year} to {end_year}) ---")
    model_datasets = {
        "TerraClimate": "IDAHO_EPSCOR/TERRACLIMATE",
        "CFSv2": "NOAA/CFSV2/FOR6H_HARMONIZED",
        "DynamicWorld": "GOOGLE/DYNAMICWORLD/V1"
    }
    
    for name, dataset_id in model_datasets.items():
        try:
            aoi_ee = build_aoi(gdf, dataset_id)
            for i in range(start_year, end_year + 1):
                export_gee_dataset(
                    dataset_id=dataset_id,
                    aoi=aoi_ee,
                    start_date=f"{i}-01-01",
                    end_date=f"{i + 1}-01-01",
                    filename_prefix=f"{name}_{aoi_name}_{i}",
                    export_format="CSV",
                    folder=drive_folder,
                )
        except Exception as e:
            print(f"Failed to initialize {name}: {e}")

    print(f"\n All Earth Engine tasks have been submitted! Check your Google Drive folder: '{drive_folder}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and export GEE data to Google Drive using PostGIS AOIs.")
    parser.add_argument("--aoi", type=str, required=True, help="Name of the AOI in the PostGIS database (e.g., test_site_one)")
    parser.add_argument("--folder", type=str, default="GEE_Morocco_Forests", help="Name of the folder in Google Drive to save CSVs")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year for time series data")
    parser.add_argument("--end-year", type=int, default=2023, help="End year for time series data")
    
    args = parser.parse_args()
    main(args.aoi, args.folder, args.start_year, args.end_year)
