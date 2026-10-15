import os
import sys
from pathlib import Path

# Unified Google Drive scope
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """
    Authenticates with Google Drive API and returns a service object.
    Supports GCP Service Account (ADC) on VM / Cloud Run, and OAuth token/credentials locally.
    """
    # 1. Attempt Application Default Credentials (ADC) / GCP Service Account (VM / Cloud Run)
    try:
        import google.auth
        from googleapiclient.discovery import build
        creds, _ = google.auth.default(scopes=SCOPES)
        if creds:
            service = build('drive', 'v3', credentials=creds)
            service.files().list(pageSize=1, fields="files(id)").execute()
            return service
    except Exception:
        pass

    # 2. Attempt token.json (User OAuth credentials)
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_file = 'token.json'
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            if creds and creds.valid:
                return build('drive', 'v3', credentials=creds)
        except Exception:
            pass

    # 3. Interactive flow via credentials.json (if run in terminal)
    if os.path.exists('credentials.json') and sys.stdin.isatty():
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            print("  [Drive] Requesting Google Drive write permissions via browser...")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
            return build('drive', 'v3', credentials=creds)
        except Exception as e:
            print(f"  [Drive] Notice: Interactive OAuth skipped: {e}")
            return None

    return None

def get_or_create_drive_folder(service, folder_name: str) -> str:
    """Searches Drive for a folder by name, or creates it if not found."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']

    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"  [Drive] Created Google Drive folder '{folder_name}' (ID: {folder_id})")
    return folder_id

def upload_file_to_drive(file_path: Path, drive_folder_name: str = "GEE_Morocco_Forests", overwrite: bool = False):
    """Uploads a CSV file to the specified Google Drive folder if not already present."""
    try:
        from googleapiclient.http import MediaFileUpload
        service = get_drive_service()
        if not service:
            print(f"  [Drive] Notice: Drive credentials not configured. File remains locally at {file_path.name}")
            return None

        folder_id = get_or_create_drive_folder(service, drive_folder_name)
        file_name = file_path.name

        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing = results.get('files', [])

        if existing:
            file_id = existing[0]['id']
            if not overwrite:
                print(f"  [Drive] '{file_name}' already in Drive folder '{drive_folder_name}' (skipping upload).")
                return file_id
            media = MediaFileUpload(str(file_path), mimetype='text/csv', resumable=True)
            service.files().update(fileId=file_id, media_body=media).execute()
            print(f"  [Drive] Updated '{file_name}' in Drive folder '{drive_folder_name}'")
            return file_id
        else:
            media = MediaFileUpload(str(file_path), mimetype='text/csv', resumable=True)
            metadata = {'name': file_name, 'parents': [folder_id]}
            uploaded = service.files().create(body=metadata, media_body=media, fields='id').execute()
            file_id = uploaded.get('id')
            print(f"  [Drive] Uploaded '{file_name}' to Drive folder '{drive_folder_name}' (ID: {file_id})")
            return file_id
    except Exception as e:
        print(f"  [Drive] Warning: Could not upload to Drive ({e}). File remains locally at {file_path.name}")
        return None
