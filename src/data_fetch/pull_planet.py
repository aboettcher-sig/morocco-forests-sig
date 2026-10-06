import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import geopandas as gpd
import requests
from shapely.geometry import mapping
from shapely.ops import unary_union

# Add the parent directory to the system path so we can import db_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database")))
from db_utils import get_db_engine

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BASE_URL = "https://api.planet.com/basemaps/v1"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
END_STATES = {"success", "partial", "failed", "cancelled"}
POLL_SECONDS = 10

def get_session():
    """Initializes and returns a requests session with Planet API authentication."""
    api_key = os.getenv("PLANET_API_KEY")
    if not api_key or api_key == "your_planet_api_key_here":
        raise ValueError("PLANET_API_KEY is missing or invalid in the .env file.")
    session = requests.Session()
    session.auth = (api_key, "")
    return session

def _get(session, url, params=None, stream=False):
    for attempt in range(6):
        r = session.get(url, params=params, stream=stream, timeout=180)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if not r.ok:
            raise RuntimeError(f"GET {url} -> {r.status_code} {r.text[:500]}")
        return r
    raise RuntimeError(f"GET {url} -> {r.status_code} {r.text[:500]}")

def _post(session, url, payload):
    for attempt in range(6):
        r = session.post(url, json=payload, timeout=180)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if not r.ok:
            raise RuntimeError(f"POST {url} -> {r.status_code} {r.text[:500]}")
        return r
    raise RuntimeError(f"POST {url} -> {r.status_code} {r.text[:500]}")

def paginate(session, url, item_key, params=None):
    while url:
        payload = _get(session, url, params=params).json()
        params = None
        for item in payload.get(item_key, []):
            yield item
        url = payload.get("_links", {}).get("_next")

def fetch_aoi_from_db(aoi_name: str) -> dict:
    """Connects to PostGIS, fetches the AOI by name, and returns it as a GeoJSON geometry."""
    engine = get_db_engine()
    # Security: Use parameterized query to prevent SQL injection
    query = "SELECT name, geometry FROM aoi_boundaries WHERE name = %(name)s"
    
    print(f"Fetching AOI '{aoi_name}' from PostGIS...")
    gdf = gpd.read_postgis(query, con=engine, geom_col='geometry', params={"name": aoi_name})
    
    if gdf.empty:
        raise ValueError(f"AOI '{aoi_name}' not found in the database. Did you load it first?")
    
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    aoi_geom = unary_union(gdf.geometry.values)
    aoi_geojson = mapping(aoi_geom)
    
    print(f"Found AOI: {len(gdf)} features, geometry type: {aoi_geojson['type']}")
    return aoi_geojson

def order_payload(mosaic_name: str, aoi_geojson: dict):
    return {
        "name": f"weekly_clip_{mosaic_name}",
        "source_type": "basemaps",
        "order_type": "partial",
        "products": [{"mosaic_name": mosaic_name, "geometry": aoi_geojson}],
        "tools": [{"merge": {}}, {"clip": {}}],
    }

def poll_order(session, order_id):
    url = f"{ORDERS_URL}/{order_id}"
    while True:
        state = _get(session, url).json()["state"]
        if state in END_STATES:
            return _get(session, url).json()
        time.sleep(POLL_SECONDS)

def download_results(session, order_json, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for r in order_json.get("_links", {}).get("results", []):
        loc = r.get("location")
        if not loc:
            continue
        name = (r.get("name") or loc.split("?")[0]).split("/")[-1]
        dest = dest_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Bypass Windows 260 character MAX_PATH limit
        dest_str = str(dest.absolute())
        if os.name == 'nt' and not dest_str.startswith('\\\\?\\'):
            dest_str = '\\\\?\\' + dest_str
            
        with _get(session, loc, stream=True) as resp, open(dest_str, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        saved.append(dest)
    return saved

def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def main(aoi_name, out_dir, series_filter, start_date, end_date):
    session = get_session()
    
    # 1. Get the AOI directly from our PostGIS database
    aoi_geojson = fetch_aoi_from_db(aoi_name)
    
    # 2. Find matching Planet Basemap Series
    all_series = list(paginate(session, f"{BASE_URL}/series/", "series"))
    matches = [s for s in all_series if series_filter.lower() in s.get("name", "").lower()]
    if not matches:
        raise RuntimeError(f"No series matched '{series_filter}'.")
    series = matches[0]
    print(f"Matched Series: {series['name']} ({series['id']})")

    # 3. Find the Mosaics within the timeframe
    links = series.get("_links", {})
    mosaics_url = links.get("mosaics") or next(
        (v for k, v in links.items() if isinstance(v, str) and "/mosaics" in v and k != "_self"),
        f"{BASE_URL}/series/{series['id']}/mosaics/",
    )
    mosaics = list(paginate(session, mosaics_url, "mosaics"))
    ordered_mosaics = sorted(mosaics, key=lambda m: m["first_acquired"])
    
    lo = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc) if start_date else datetime.min.replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) if end_date else datetime.max.replace(tzinfo=timezone.utc)
    
    selected = [m for m in ordered_mosaics if lo <= _parse(m["first_acquired"]) < hi]
    
    if not selected:
        print(f"No mosaics found between {start_date} and {end_date}.")
        return

    print(f"Ordering {len(selected)} weeks of data...")
    
    # 4. Request the clip and download the results
    made = []
    out_path = Path(out_dir)
    for m in selected:
        name = m["name"]
        print(f"Ordering {name} ...", flush=True)
        try:
            order_id = _post(session, ORDERS_URL, order_payload(name, aoi_geojson)).json()["id"]
        except RuntimeError as e:
            if "no basemap quads were found" in str(e):
                print("  Skipped: No satellite coverage for this specific week over your AOI.")
                continue
            else:
                raise e
        
        final = poll_order(session, order_id)
        
        if final["state"] in {"failed", "cancelled"}:
            print(f"  {final['state']}: {name} (order {order_id})")
            continue
            
        files = download_results(session, final, out_path / name)
        made += files
        print(f"  Downloaded: {len(files)} file(s) -> {out_dir}/{name}")

    print(f"\nDone! Saved {len(made)} files to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and clip Planet weekly basemaps using PostGIS AOIs.")
    parser.add_argument("--aoi", type=str, required=True, help="Name of the AOI in the PostGIS database (e.g., test_site_one)")
    parser.add_argument("--out-dir", type=str, default="data/planet_imagery", help="Local directory to save the downloaded GeoTIFFs")
    parser.add_argument("--series", type=str, default="weekly", help="Substring to match the Planet series name (default: 'weekly')")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    main(args.aoi, args.out_dir, args.series, args.start, args.end)
