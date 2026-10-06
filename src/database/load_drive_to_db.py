import os
import sys
import io
import tempfile
import argparse
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine
from sqlalchemy import text

# Import the categorizer we built in the local script
from load_gee_csv import get_table_name

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """Authenticates with the Google Drive API and returns a service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(host='127.0.0.1', port=8080)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('drive', 'v3', credentials=creds)

def find_folder_id(service, folder_name):
    """Searches Drive for a folder by name and returns its ID."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        return None
    return items[0]['id']

def download_and_load(folder_name):
    print("Connecting to Google Drive API...")
    try:
        service = authenticate_drive()
    except FileNotFoundError:
        print(" Error: 'credentials.json' not found. You must download it from the Google Cloud Console and place it in this folder.")
        return
        
    folder_id = find_folder_id(service, folder_name)
    if not folder_id:
        print(f" Error: Could not find folder '{folder_name}' in Google Drive.")
        return
        
    # Get all CSV files inside the folder
    query = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, size)", pageSize=1000).execute()
    files = results.get('files', [])
    
    if not files:
        print(f" No CSV files found in folder '{folder_name}'.")
        return
        
    print(f" Found {len(files)} CSV files in Google Drive. Streaming directly into PostGIS...")
    engine = get_db_engine()
    
    for file in files:
        table_name = get_table_name(file['name'])
        
        # Check if already ingested
        skip_file = False
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT 1 FROM {table_name} WHERE source_file = '{file['name']}' LIMIT 1"))
                if result.fetchone():
                    skip_file = True
        except Exception:
            pass  # Table might not exist yet
            
        if skip_file:
            print(f"  ⏭️  Skipping {file['name']} ➡️  Table: {table_name} (already ingested)")
            continue
            
        print(f"  Streaming {file['name']} ➡️  Table: {table_name}")
        
        try:
            # Download file to a temporary file on disk (Saves RAM for massive files)
            request = service.files().get_media(fileId=file['id'])
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                downloader = MediaIoBaseDownload(temp_file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                temp_file_path = temp_file.name

            # Process the CSV in manageable chunks to prevent Out-Of-Memory (OOM) crashes
            chunk_size = 50000
            first_chunk = True
            
            for chunk in pd.read_csv(temp_file_path, chunksize=chunk_size):
                chunk['source_file'] = file['name']
                
                # Truncate column names to 63 characters to match PostgreSQL limits
                chunk.columns = [str(c)[:63] for c in chunk.columns]
                
                if first_chunk:
                    # Align schema dynamically (only need to check once per file)
                    with engine.begin() as conn:
                        result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"))
                        existing_cols = {row[0] for row in result}
                        if existing_cols:
                            for col in chunk.columns:
                                if col.lower() not in existing_cols and col not in existing_cols:
                                    ctype = "DOUBLE PRECISION" if pd.api.types.is_numeric_dtype(chunk[col]) else "TEXT"
                                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {ctype}'))
                                    print(f"    [Schema] Added missing column '{col}' to {table_name}")
                    first_chunk = False
                                    
                # Push the chunk to PostgreSQL
                chunk.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
                
            # Cleanup the temporary file from the hard drive
            os.remove(temp_file_path)
            
        except Exception as e:
            error_str = str(e)
            if len(error_str) > 500:
                print(f"   Failed to process {file['name']}: {error_str[:250]} ... [truncated]")
                if hasattr(e, "orig"):
                    print(f"     Original Error: {e.orig}")
            else:
                print(f"   Failed to process {file['name']}: {e}")
            
    print("\n All Google Drive data successfully streamed into PostGIS!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream GEE CSVs directly from Google Drive into PostGIS.")
    parser.add_argument("--folder", type=str, required=True, help="Name of the Google Drive folder (e.g., Morocco_Forests_5Yr)")
    args = parser.parse_args()
    
    download_and_load(args.folder)
