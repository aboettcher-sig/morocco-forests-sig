import os
import sys
import argparse
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Connect to our robust db_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine

def get_table_name(filename):
    """Intelligently categorizes the CSVs into different PostGIS tables based on the satellite/model."""
    name = filename.lower()
    if 'landsat' in name:
        return 'gee_landsat_data'
    elif 'sentinel1' in name or 'sentinel_1' in name or 's1' in name:
        return 'gee_sentinel1_data'
    elif 'sentinel2' in name or 'sentinel_2' in name or 'sentinel' in name:
        return 'gee_sentinel2_data'
    elif 'terraclimate' in name or 'cfsv2' in name:
        return 'gee_climate_data'
    elif 'srtm' in name or 'terrain' in name:
        return 'gee_terrain_data'
    elif 'dynamicworld' in name:
        return 'gee_dynamicworld_data'
    else:
        return 'gee_other_data'

def is_file_ingested(source_file: str, engine=None) -> bool:
    """Checks if a file has already been ingested into PostGIS."""
    if engine is None:
        engine = get_db_engine()
    table_name = get_table_name(source_file)
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            res = conn.execute(text(f"SELECT 1 FROM {table_name} WHERE source_file = :src LIMIT 1"), {"src": source_file})
            return res.fetchone() is not None
    except Exception:
        return False

def ingest_df_to_db(df: pd.DataFrame, source_file: str, engine=None) -> bool:
    """Ingests a DataFrame into PostGIS if not already loaded, preventing duplicates."""
    if engine is None:
        engine = get_db_engine()

    table_name = get_table_name(source_file)
    if is_file_ingested(source_file, engine):
        print(f"  [Skip] {source_file} -> {table_name} (already ingested)")
        return True

    try:
        clean_df = df.copy()
        clean_df.columns = [str(col).replace(':', '_').replace('.', '_') for col in clean_df.columns]
        clean_df['source_file'] = source_file

        # Dynamically align table schema if new satellite metadata bands appear
        with engine.begin() as conn:
            from sqlalchemy import text
            res = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"))
            existing_cols = {row[0].lower() for row in res}
            if existing_cols:
                for col in clean_df.columns:
                    if col.lower() not in existing_cols:
                        ctype = "DOUBLE PRECISION" if pd.api.types.is_numeric_dtype(clean_df[col]) else "TEXT"
                        conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {ctype}'))
                        print(f"  [Schema] Added missing column '{col}' to '{table_name}'")

        print(f"  [PostgreSQL] Ingesting {len(clean_df)} rows into '{table_name}'...")
        clean_df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print(f"  [PostgreSQL] Done! Ingested into {table_name}")
        return True
    except Exception as e:
        print(f"  [PostgreSQL] Failed to load {source_file}: {e}")
        return False

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
        if is_file_ingested(file.name, engine):
            print(f"  [Skip] {file.name} -> {table_name} (already ingested)")
            continue
        try:
            df = pd.read_csv(file)
            ingest_df_to_db(df, file.name, engine)
        except Exception as e:
            print(f"   Failed to read {file.name}: {e}")
            
    print("\n All Earth Engine data successfully loaded into your PostGIS database!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load exported Earth Engine CSVs into PostGIS.")
    parser.add_argument("--csv-dir", type=str, required=True, help="Local path to the folder containing the downloaded CSVs")
    
    args = parser.parse_args()
    load_csv_to_db(args.csv_dir)
