import os
import argparse
import pandas as pd
from pathlib import Path
import sys

# Connect to our robust db_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine

def get_table_name(filename):
    """Intelligently categorizes the CSVs into different PostGIS tables based on the satellite/model."""
    name = filename.lower()
    if 'landsat' in name:
        return 'gee_landsat_data'
    elif 'sentinel' in name:
        return 'gee_sentinel_data'
    elif 'terraclimate' in name or 'cfsv2' in name:
        return 'gee_climate_data'
    elif 'srtm' in name or 'terrain' in name:
        return 'gee_terrain_data'
    elif 'dynamicworld' in name:
        return 'gee_dynamicworld_data'
    else:
        return 'gee_other_data'

def load_csv_to_db(csv_dir):
    engine = get_db_engine()
    csv_path = Path(csv_dir)
    
    if not csv_path.exists():
        print(f" Error: Directory '{csv_dir}' does not exist.")
        return

    csv_files = list(csv_path.glob("*.csv"))
    if not csv_files:
        print(f" No CSV files found in '{csv_dir}'.")
        return

    print(f" Found {len(csv_files)} CSV files. Beginning ingestion into PostGIS...")
    
    for file in csv_files:
        table_name = get_table_name(file.name)
        print(f"  Loading {file.name} ➡️  Table: {table_name}")
        
        try:
            # Read the CSV
            df = pd.read_csv(file)
            
            # Add a metadata column so you always know which file this data came from
            df['source_file'] = file.name
            
            # Push directly to PostgreSQL in chunks (handles massive datasets without crashing)
            df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
            
        except Exception as e:
            print(f"   Failed to load {file.name}: {e}")
            
    print("\n All Earth Engine data successfully loaded into your PostGIS database!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load exported Earth Engine CSVs into PostGIS.")
    parser.add_argument("--csv-dir", type=str, required=True, help="Local path to the folder containing the downloaded CSVs")
    
    args = parser.parse_args()
    load_csv_to_db(args.csv_dir)
