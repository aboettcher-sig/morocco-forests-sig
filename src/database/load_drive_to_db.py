import os
import sys
import tempfile
import argparse
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_fetch")))
from db_utils import get_db_engine
from load_gee_csv import get_table_name, is_file_ingested
from drive_utils import get_drive_service
from sqlalchemy import text


def find_all_folder_ids(service, folder_name: str):
    """Searches Drive for all folders with the given name and returns a set of their IDs."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)", pageSize=1000).execute()
    return [f['id'] for f in results.get('files', [])]


def download_and_load(folder_name: str):
    print(f"Connecting to Google Drive API...")
    service = get_drive_service()
    if not service:
        print(" Error: Google Drive credentials not found or unauthenticated.")
        return

    folder_ids = find_all_folder_ids(service, folder_name)
    if not folder_ids:
        print(f" Error: Could not find any folder named '{folder_name}' in Google Drive.")
        return

    print(f" Found {len(folder_ids)} folder instance(s) named '{folder_name}'. Collecting CSV files...")
    
    # Query all CSV files in Drive that belong to any of these folders
    files = []
    # Batch query in groups of 20 parents to prevent query length overflow
    for i in range(0, len(folder_ids), 20):
        parent_group = folder_ids[i:i+20]
        parents_clause = " or ".join([f"'{fid}' in parents" for fid in parent_group])
        query = f"({parents_clause}) and mimeType='text/csv' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, size)", pageSize=1000).execute()
        files.extend(results.get('files', []))

    if not files:
        print(f" No CSV files found in '{folder_name}' folders.")
        return

    print(f" Found {len(files)} CSV files in Google Drive. Streaming directly into PostGIS...")
    engine = get_db_engine()

    for file in files:
        table_name = get_table_name(file['name'])

        if is_file_ingested(file['name'], engine):
            print(f"  [Skip] {file['name']} -> {table_name} (already ingested)")
            continue

        print(f"  Streaming {file['name']} -> {table_name}")

        try:
            # Download file to a temporary file on disk
            request = service.files().get_media(fileId=file['id'])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                downloader = MediaIoBaseDownload(temp_file, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                temp_file_path = temp_file.name

            # Process the CSV in manageable chunks
            chunk_size = 50000
            first_chunk = True

            for chunk in pd.read_csv(temp_file_path, chunksize=chunk_size):
                chunk['source_file'] = file['name']
                chunk.columns = [str(c).replace(':', '_').replace('.', '_')[:63] for c in chunk.columns]

                if first_chunk:
                    with engine.begin() as conn:
                        res = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"))
                        existing_cols = {row[0] for row in res}
                        if existing_cols:
                            for col in chunk.columns:
                                if col.lower() not in existing_cols and col not in existing_cols:
                                    ctype = "DOUBLE PRECISION" if pd.api.types.is_numeric_dtype(chunk[col]) else "TEXT"
                                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {ctype}'))
                                    print(f"    [Schema] Added missing column '{col}' to {table_name}")
                    first_chunk = False

                chunk.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)

            os.remove(temp_file_path)

        except Exception as e:
            print(f"   Failed to process {file['name']}: {e}")

    print("\n All Google Drive data successfully streamed into PostGIS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream GEE CSVs directly from Google Drive into PostGIS.")
    parser.add_argument("--folder", type=str, required=True, help="Name of the Google Drive folder")
    args = parser.parse_args()

    download_and_load(args.folder)
