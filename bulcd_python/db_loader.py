"""Build the per-pixel NBR bin series from the FAO PostGIS database.

Replaces afn_gatherCollectionsAndReduce (515.gatherCollections27b) for the
optical sensors used by the caller (S2, L8, L9):

* Landsat 8/9: QA_PIXEL bits 0-4 clear and QA_RADSAT == 0 (maskSrCloudsL8andL9),
  SR scaling 0.0000275 / -0.2 (applyScaleFactorsL789), NBR = (SR_B5-SR_B7)/(SR_B5+SR_B7).
* Sentinel-2: NBR = (B8-B12)/(B8+B12). GEE masked clouds with Cloud Score+
  (cs >= 0.6), which is not in the database; here SCL in (4, 5, 6) and
  MSK_CLDPRB < 10 is used instead (same rule as point_timeseries.py).
* NBR > 0.8 is masked for every sensor.
* Each 2-day bin is the median of every clear observation from every sensor.

Pixels are 10 m Sentinel-2 sub-pixels (cell_id, subpixel_id 0-8, row-major from
the north-west corner). Landsat is sampled once per 30 m cell, so its value is
shared by the 9 sub-pixels, like nearest-neighbour resampling in GEE.
"""
from __future__ import annotations

import getpass
import hashlib
import io
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2
import shapely
from shapely import wkt as shp_wkt

from bulcd_core import Calendar

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env_files() -> None:
    """Read KEY=VALUE lines from .env next to this file, then the repo root's.

    Variables already set in the environment win. No python-dotenv needed.
    """
    for path in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.removeprefix("export ").strip()
                os.environ.setdefault(key, val.strip().strip("'\""))


_load_env_files()
# Connection settings come from the environment or a .env file (never committed):
# POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD.
DB = dict(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", "5432")),
    dbname=os.environ.get("POSTGRES_DB", "geo_db"),
    user=os.environ.get("POSTGRES_USER", "geo_user"),
)
CACHE_DIR = os.path.join(HERE, "cache")
LANDSAT_PREFIX = {"L8": "LC08", "L9": "LC09"}
NBR_MAX = 0.8
# Years present in the database. The tables have no date index, so a query reads
# every year of its cells anyway: when a cell batch has to be downloaded, all of
# these years are fetched in the same pass and cached for later runs.
DATA_YEARS = range(2021, 2026)
_pw_lock = threading.Lock()


def _password() -> str:
    with _pw_lock:
        pw = (os.environ.get("FAO_DB_PASSWORD") or os.environ.get("PGPASSWORD")
              or os.environ.get("POSTGRES_PASSWORD"))
        if not pw:
            pw = getpass.getpass(f"Password for {DB['user']}@{DB['dbname']}: ")
            os.environ["FAO_DB_PASSWORD"] = pw
        return pw


def connect():
    return psycopg2.connect(**DB, password=_password(), connect_timeout=15)


# ---------------------------------------------------------------------------
# AOI and pixels
# ---------------------------------------------------------------------------
def aoi_sql(*, bbox=None, wkt=None, aoi_id=None, wkt_26191=None) -> tuple[str, tuple]:
    """SQL expression (EPSG:26191) for the area of interest, plus its params."""
    if wkt_26191 is not None:
        return "ST_GeomFromText(%s, 26191)", (wkt_26191,)
    if aoi_id is not None:
        return "(select geom_26191 from aoi_boundaries where aoi_id = %s)", (int(aoi_id),)
    if wkt is not None:
        return "ST_Transform(ST_GeomFromText(%s, 4326), 26191)", (wkt,)
    if bbox is not None:
        return "ST_Transform(ST_MakeEnvelope(%s, %s, %s, %s, 4326), 26191)", tuple(map(float, bbox))
    raise ValueError("give bbox, wkt or aoi_id")


def load_pixels(conn, aoi: tuple[str, tuple]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Cells intersecting the AOI and their 9 sub-pixels.

    Returns (cells, pixels, aoi_wkt_26191). pixels has one row per sub-pixel
    (index = cell_position * 9 + subpixel_id) and an `in_aoi` flag for
    sub-pixels whose centre lies inside the AOI (GEE clips to the AOI).
    """
    expr, params = aoi
    cur = conn.cursor()
    cur.execute(f"select ST_AsText({expr})", params)
    row = cur.fetchone()
    if not row or not row[0]:
        raise SystemExit("AOI not found")
    aoi_wkt = row[0]
    cells = pd.read_sql(
        f"""select cell_id, box_id, ST_XMin(geom) xmin, ST_YMin(geom) ymin, ST_XMax(geom) xmax, ST_YMax(geom) ymax
            from grid_30m_morocco where ST_Intersects(geom, {expr}) order by cell_id""",
        conn, params=params)
    if cells.empty:
        raise SystemExit("No grid cells intersect the AOI")
    poly = shp_wkt.loads(aoi_wkt)
    shapely.prepare(poly)

    def subpixels(cells):
        sub = np.tile(np.arange(9), len(cells))
        rep = cells.loc[cells.index.repeat(9)].reset_index(drop=True)
        size = (rep.xmax - rep.xmin) / 3
        x = rep.xmin + (sub % 3 + 0.5) * size
        y = rep.ymax - (sub // 3 + 0.5) * size
        return sub, rep, size, x, y, shapely.covers(poly, shapely.points(x, y))

    sub, rep, size, x, y, inside = subpixels(cells)
    # cells that only touch the AOI (no sub-pixel centre inside) are not needed
    cells = cells[pd.Series(inside).groupby(np.repeat(np.arange(len(cells)), 9)).any().to_numpy()].reset_index(drop=True)
    sub, rep, size, x, y, inside = subpixels(cells)
    pixels = pd.DataFrame({"cell_id": rep.cell_id, "subpixel_id": sub, "box_id": rep.box_id,
                           "x": x, "y": y, "size": size, "in_aoi": inside})
    return cells, pixels, aoi_wkt


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
# Cache: cache/index.json lists "batches" of up to 300 cells. A batch has one
# sensor kind ("S2" or "L" = Landsat), the signature of the query that made it
# (so changing a mask invalidates it), its cell ids, and one .npz file per
# calendar year with (cell_id, sub, day-of-year from Jan 1 = 0, nbr).
# Any AOI reuses the cells already cached, whatever batches they came from.
# The database only indexes cell_id (not date): every query reads all years
# of its cells, so missing cells are downloaded for all DATA_YEARS at once.
INDEX = os.path.join(CACHE_DIR, "index.json")
_S2_SELECT = """select cell_id, subpixel_id, left(date, 4)::int, (date::date - make_date(left(date, 4)::int, 1, 1)),
                       round((("B8" - "B12") / nullif("B8" + "B12", 0))::numeric, 5)
                from gee_sentinel2_data
                where cell_id = any(%s) and left(date, 4) = any(%s)
                  and "SCL" in (4, 5, 6) and "MSK_CLDPRB" < 10"""
_LANDSAT_SELECT = """select cell_id, -1, left(date, 4)::int, (date::date - make_date(left(date, 4)::int, 1, 1)),
                       round(((n - s) / nullif(n + s, 0))::numeric, 5)
                from (select cell_id, date,
                             "SR_B5" * 0.0000275 - 0.2 n, "SR_B7" * 0.0000275 - 0.2 s
                      from gee_landsat_data
                      where cell_id = any(%s) and left(date, 4) = any(%s)
                        and left(image_id, 4) = any(%s)
                        and ("QA_PIXEL"::bigint & 31) = 0 and "QA_RADSAT" = 0) q"""


_WATER_SELECT = """select cell_id, subpixel_id, count(*) filter (where "SCL" = 6), count(*)
                   from gee_sentinel2_data
                   where cell_id = any(%s) and left(date, 4) = any(%s)
                     and "SCL" in (4, 5, 6) and "MSK_CLDPRB" < 10
                   group by 1, 2"""
WATER_DIR = os.path.join(CACHE_DIR, "water")


def _water_batch(cell_ids, years, retries=6):
    yrs = [str(y) for y in years]
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            cur.execute(_WATER_SELECT, (cell_ids, yrs))
            rows = cur.fetchall()
            break
        except (psycopg2.OperationalError, psycopg2.DatabaseError, OSError) as e:
            if attempt == retries:
                raise
            wait = min(30 * 2 ** attempt, 300)
            print(f"  water batch failed ({type(e).__name__}); retry {attempt + 1}/{retries} in {wait} s", flush=True)
            time.sleep(wait)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    a = np.array(rows, dtype=np.int64).reshape(-1, 4)
    return a


def water_counts(cells: pd.DataFrame, years=DATA_YEARS, *, chunk_size=300, workers=3, log=print) -> pd.DataFrame:
    """Per Sentinel-2 sub-pixel: clear observations (SCL 4/5/6, cloud prob < 10) and how many of
    them SCL classes as water (6). Cached in cache/water/ per batch of cells."""
    os.makedirs(WATER_DIR, exist_ok=True)
    idx_path = os.path.join(WATER_DIR, "index.json")
    sig = hashlib.sha1((_WATER_SELECT + "|" + ",".join(map(str, years))).encode()).hexdigest()[:12]
    index = json.load(open(idx_path))["batches"] if os.path.exists(idx_path) else []
    mine = [b for b in index if b["sig"] == sig]
    have = set().union(*[b["cells"] for b in mine]) if mine else set()
    miss = sorted(set(int(c) for c in cells.cell_id) - have)
    log(f"  water (SCL): {len(cells) - len(miss)} of {len(cells)} cells cached, {len(miss)} to query")
    jobs = [miss[i:i + chunk_size] for i in range(0, len(miss), chunk_size)]
    if jobs:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_water_batch, ids, list(years)): ids for ids in jobs}
            for done, f in enumerate(as_completed(futs), 1):
                ids = futs[f]
                name = "water_" + hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:16] + ".npy"
                np.save(os.path.join(WATER_DIR, name), f.result())
                index.append({"sig": sig, "cells": [int(c) for c in ids], "file": name})
                with open(idx_path + ".tmp", "w") as fh:
                    json.dump({"batches": index}, fh)
                os.replace(idx_path + ".tmp", idx_path)
                if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                    el = time.time() - t0
                    log(f"  water: {done}/{len(jobs)} batches; {el / 60:.0f} min, about {el / done * (len(jobs) - done) / 60:.0f} min left")
    want = set(int(c) for c in cells.cell_id)
    parts = [np.load(os.path.join(WATER_DIR, b["file"])) for b in index
             if b["sig"] == sig and want.intersection(b["cells"])]
    a = np.concatenate(parts) if parts else np.zeros((0, 4), np.int64)
    df = pd.DataFrame(a, columns=["cell_id", "subpixel_id", "n_water", "n_clear"])
    return df[df.cell_id.isin(want)].drop_duplicates(["cell_id", "subpixel_id"], keep="last")


def query_signature(kind: str, prefixes) -> str:
    text = _S2_SELECT if kind == "S2" else _LANDSAT_SELECT + "|" + ",".join(sorted(prefixes))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def load_index() -> list[dict]:
    if not os.path.exists(INDEX):
        return []
    with open(INDEX) as f:
        return json.load(f)["batches"]


def save_index(batches: list[dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(INDEX + ".tmp", "w") as f:
        json.dump({"batches": batches}, f)
    os.replace(INDEX + ".tmp", INDEX)


def _to_obs(df: pd.DataFrame) -> np.ndarray:
    return np.rec.fromarrays([df.cell_id.to_numpy(np.int64), df["sub"].to_numpy(np.int8),
                              df.day.to_numpy(np.int16), df.nbr.to_numpy(np.float32)],
                             names="cell_id,sub,day,nbr")


def _query_batch(kind, cell_ids, years, prefixes) -> io.StringIO:
    conn = connect()
    try:
        cur = conn.cursor()
        yrs = [str(y) for y in years]
        sql = (cur.mogrify(f"copy ({_S2_SELECT}) to stdout with csv", (cell_ids, yrs)) if kind == "S2" else
               cur.mogrify(f"copy ({_LANDSAT_SELECT}) to stdout with csv", (cell_ids, yrs, list(prefixes)))).decode()
        buf = io.StringIO()
        cur.copy_expert(sql, buf)
        return buf
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _download_batch(kind: str, sig: str, cell_ids: list[int], years: list[int], prefixes,
                    retries: int = 6) -> dict:
    """One database query for a batch of cells and all `years`; writes one file per year.

    Connection errors are retried with a growing wait (30 s, 60 s, ... up to 5 min)."""
    for attempt in range(retries + 1):
        try:
            buf = _query_batch(kind, cell_ids, years, prefixes)
            break
        except (psycopg2.OperationalError, psycopg2.DatabaseError, OSError) as e:
            if attempt == retries:
                raise
            wait = min(30 * 2 ** attempt, 300)
            print(f"  batch of {len(cell_ids)} {kind} cells failed ({type(e).__name__}: "
                  f"{str(e).strip()[:80]}); retry {attempt + 1}/{retries} in {wait} s", flush=True)
            time.sleep(wait)
    buf.seek(0)
    df = pd.read_csv(buf, header=None, names=["cell_id", "sub", "year", "day", "nbr"])
    bid = hashlib.sha1(f"{kind}|{sig}|{','.join(map(str, cell_ids))}".encode()).hexdigest()[:16]
    files = {}
    for y in years:
        name = f"{kind}_{bid}_{y}.npz"
        path = os.path.join(CACHE_DIR, name)
        np.savez_compressed(path + ".tmp.npz", obs=_to_obs(df[df.year == y]))
        os.replace(path + ".tmp.npz", path)
        files[str(y)] = name
    return {"kind": kind, "sig": sig, "cells": [int(c) for c in cell_ids], "files": files,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _expand_landsat(obs, n_sub=9):
    """Copy each Landsat cell observation to the 9 sub-pixels."""
    rep = np.repeat(obs, n_sub)
    rep["sub"] = np.tile(np.arange(n_sub, dtype=np.int8), len(obs))
    return rep


def ensure_cached(cells: pd.DataFrame, years, sensors, *, chunk_size=300, workers=3, refresh=False, log=print):
    """Download the cells missing from the cache for any of `years` (all DATA_YEARS at once)."""
    sensors = set(sensors)
    unknown = sensors - {"S2", "L8", "L9"}
    if unknown:
        raise NotImplementedError(f"sensors not in the database port: {sorted(unknown)}")
    prefixes = [LANDSAT_PREFIX[s] for s in ("L8", "L9") if s in sensors]
    kinds = (["S2"] if "S2" in sensors else []) + (["L"] if prefixes else [])
    fetch_years = sorted(set(years) | set(DATA_YEARS))
    needed = set(int(c) for c in cells.cell_id)
    os.makedirs(CACHE_DIR, exist_ok=True)
    index = load_index()
    usable = [] if refresh else index
    jobs = []
    for kind in kinds:
        sig = query_signature(kind, prefixes)
        mine = [b for b in usable if b["kind"] == kind and b["sig"] == sig]
        missing = set()
        for y in years:
            have = set().union(*[b["cells"] for b in mine if str(y) in b["files"]]) if mine else set()
            missing |= needed - have
        miss = sorted(missing)
        log(f"  {kind}: {len(needed) - len(miss)} of {len(needed)} cells cached, {len(miss)} to download")
        jobs += [(kind, sig, miss[i:i + chunk_size]) for i in range(0, len(miss), chunk_size)]
    if jobs:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_download_batch, k, sg, ids, fetch_years, prefixes) for k, sg, ids in jobs]
            for done, f in enumerate(as_completed(futs), 1):
                index.append(f.result())
                save_index(index)
                if done % max(1, len(jobs) // 40) == 0 or done == len(jobs):
                    el = time.time() - t0
                    log(f"  downloaded {done}/{len(jobs)} batches of up to {chunk_size} cells "
                        f"(years {fetch_years[0]}-{fetch_years[-1]}); {el / 60:.0f} min, "
                        f"about {el / done * (len(jobs) - done) / 60:.0f} min left")
    return index, kinds, prefixes


def build_series(cells: pd.DataFrame, calendars: list[Calendar], sensors, *,
                 chunk_size=300, workers=3, refresh=False, log=print) -> list[np.ndarray]:
    """One (n_pixels, calendar.n) array of bin-median NBR per calendar.

    Pixel order matches load_pixels: cell position * 9 + subpixel_id.
    """
    sensors = set(sensors)
    unknown = sensors - {"S2", "L8", "L9"}
    if unknown:
        raise NotImplementedError(f"sensors not in the database port: {sorted(unknown)}")
    prefixes = [LANDSAT_PREFIX[s] for s in ("L8", "L9") if s in sensors]
    kinds = (["S2"] if "S2" in sensors else []) + (["L"] if prefixes else [])

    d0 = min(c.fetch_range[0] for c in calendars)
    d1 = max(c.fetch_range[1] for c in calendars)
    years = list(range(d0.year, (d1 - timedelta(days=1)).year + 1))
    fetch_years = sorted(set(years) | set(DATA_YEARS))
    needed = set(int(c) for c in cells.cell_id)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Download the cells not cached yet (for every year needed).
    index, kinds, prefixes = ensure_cached(cells, years, sensors, chunk_size=chunk_size, workers=workers,
                                           refresh=refresh, log=log)
    needed = set(int(c) for c in cells.cell_id)

    # 2. Read each (sensor, year) once per cell: newest batches first, so a cell
    #    cached twice is never counted twice.
    cell_pos = pd.Series(np.arange(len(cells)), index=cells.cell_id.to_numpy())
    newest_first = sorted(index, key=lambda b: b["created"], reverse=True)
    pix_l, day_l, nbr_l = [], [], []
    for kind in kinds:
        sig = query_signature(kind, prefixes)
        for y in years:
            claimed = set()
            for b in newest_first:
                if b["kind"] != kind or b["sig"] != sig or str(y) not in b["files"]:
                    continue
                take = (set(b["cells"]) & needed) - claimed
                if not take:
                    continue
                claimed |= take
                obs = np.load(os.path.join(CACHE_DIR, b["files"][str(y)]))["obs"]
                obs = obs[np.isin(obs["cell_id"], np.fromiter(take, np.int64))]
                if kind == "L":
                    # gee_landsat_data holds exact duplicate rows (same cell, day and
                    # value) for 2021 to May 2023; GEE counts each scene once. L8 and
                    # L9 never image the same cell on the same day, so this is safe.
                    obs = np.unique(obs)
                    obs = _expand_landsat(obs)
                obs = obs[np.isfinite(obs["nbr"]) & (obs["nbr"] <= NBR_MAX)]
                pix_l.append(cell_pos.loc[obs["cell_id"]].to_numpy() * 9 + obs["sub"])
                day_l.append(obs["day"].astype(int) + (date(y, 1, 1) - d0).days)
                nbr_l.append(obs["nbr"].astype(float))
            if claimed != needed:
                raise RuntimeError(f"{kind} {y}: {len(needed - claimed)} cells missing from the cache")

    pix = np.concatenate(pix_l) if pix_l else np.array([], int)
    day = np.concatenate(day_l) if day_l else np.array([], int)
    nbr = np.concatenate(nbr_l) if nbr_l else np.array([], float)

    out = [np.full((len(cells) * 9, c.n), np.nan, dtype=np.float32) for c in calendars]
    for k, cal in enumerate(calendars):
        rel = day - (np.datetime64(cal.starts[0], "D") - np.datetime64(d0, "D")).astype(int)
        b = np.where(rel >= 0, rel // cal.step, -1)
        keep = (b >= 0) & (b < cal.n)
        df = pd.DataFrame({"pix": pix[keep], "bin": b[keep], "nbr": nbr[keep]})
        med = df.groupby(["pix", "bin"], sort=False).nbr.median()
        out[k][med.index.get_level_values(0), med.index.get_level_values(1)] = med.to_numpy()
    return out
