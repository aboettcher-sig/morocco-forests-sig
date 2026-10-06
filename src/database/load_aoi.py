import os
import argparse
import geopandas as gpd
from sqlalchemy import text, inspect
from db_utils import get_db_engine

def load_aoi_to_postgis(gpkg_path: str, table_name: str = "aoi_boundaries"):
    """
    Reads a GeoPackage boundary and loads it into a PostGIS table.
    Ensures the geometry is stored in EPSG:4326.
    """
    print(f"Reading {gpkg_path}...")
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"AOI file not found at {gpkg_path}")

    # Read the GeoPackage using GeoPandas
    gdf = gpd.read_file(gpkg_path)
    
    # Ensure it's in the standard WGS84 coordinate reference system (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting AOI from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Ensure we only have one polygon/multipolygon boundary
    if len(gdf) > 1:
        print(f"Warning: The AOI file contains {len(gdf)} features. It will be loaded as multiple rows.")

    # Name the boundary for reference
    if 'name' not in gdf.columns:
        gdf['name'] = os.path.basename(gpkg_path).replace('.gpkg', '')

    # Get unique names in the current GeoDataFrame to manage duplicates
    aoi_names = gdf['name'].unique()

    engine = get_db_engine()
    
    print(f"Uploading AOI to PostGIS table '{table_name}'...")
    
    # --- PRO SUGGESTION 1: Append instead of replace + Avoid duplicates ---
    inspector = inspect(engine)
    # Check if the table already exists
    if inspector.has_table(table_name):
        # Prevent duplicate names by deleting old records with the same name before appending
        # Using engine.begin() automatically commits the transaction
        with engine.begin() as conn:
            for name in aoi_names:
                print(f"Cleaning up old entries for AOI: '{name}' (if they exist)...")
                del_query = text(f"DELETE FROM {table_name} WHERE name = :name")
                conn.execute(del_query, {"name": name})

    # Push to PostGIS
    # 'if_exists="append"' means we add to the existing table instead of overwriting it, preserving other AOIs
    gdf.to_postgis(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        dtype={'geometry': 'Geometry'}
    )
    
    # Verify validity and area (as requested in Phase 2 validation)
    with engine.connect() as conn:
        # Check only the recently uploaded records using the IN clause
        result = conn.execute(text(f"""
            SELECT name, ST_IsValid(geometry) as is_valid, ST_Area(geometry::geography)/1e6 AS area_sq_km 
            FROM {table_name}
            WHERE name IN :names;
        """), {"names": tuple(aoi_names)}).fetchall()
        
        print("\nVerification complete. Data in database:")
        for row in result:
            print(f"   Name: {row[0]}")
            print(f"   Geometry Valid: {row[1]}")
            print(f"   Area: {row[2]:.2f} km²")

if __name__ == "__main__":
    # --- PRO SUGGESTION 2: Reusable CLI ---
    # Use argparse to allow running the script with any AOI file from the terminal
    parser = argparse.ArgumentParser(description="Load an AOI GeoPackage into the PostGIS database.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="The relative or absolute path to the .gpkg file (e.g., AOI/test_site_one.gpkg)."
    )
    parser.add_argument(
        "--table", 
        type=str, 
        default="aoi_boundaries", 
        help="The destination PostGIS table name (default: aoi_boundaries)."
    )
    
    args = parser.parse_args()
    
    # Execute the loading function using the provided CLI arguments
    load_aoi_to_postgis(args.file_path, args.table)
