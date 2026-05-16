"""Cell-availability scoping for Bird Song substrate.

Pulls full xeno-canto v3 metadata for the 4 pilot species (CONUS), fetches the
USFWS Bird Conservation Region shapefile, spatial-joins recording lat/lon
to BCR polygons, and reports per-(species x BCR) cell counts.

NO AUDIO DOWNLOAD, NO FEATURE EXTRACTION at this stage. Metadata + geography
only. Output: scoping report analogous to gun-violence cell-availability docs.
"""
import os, sys, urllib.request, urllib.parse, urllib.error, json, time, pathlib, zipfile, io
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:/Bird Song")

# Load API key from .env
for line in open(ROOT / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v
KEY = os.environ["XENOCANTO_API_KEY"]
API = "https://xeno-canto.org/api/3/recordings"

SPECIES = [
    ("Zonotrichia leucophrys",  "white-crowned sparrow"),
    ("Melospiza melodia",       "song sparrow"),
    ("Poecile carolinensis",    "Carolina chickadee"),
    ("Cistothorus palustris",   "marsh wren"),
]

# ---------- 1. Pull all metadata pages ----------
print("[1] Pulling xeno-canto metadata (paginated) ...")
all_records = []
for sci, common in SPECIES:
    q = f'sp:"{sci}" cnt:"United States"'
    page = 1
    species_recs = []
    while True:
        params = urllib.parse.urlencode({"query": q, "key": KEY, "page": page})
        url = f"{API}?{params}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        recs = data.get("recordings", [])
        species_recs.extend(recs)
        total_pages = int(data.get("numPages", 1))
        print(f"  {common:<24} page {page}/{total_pages}  +{len(recs)} (cumulative {len(species_recs)})")
        if page >= total_pages: break
        page += 1
        time.sleep(0.5)
    for r in species_recs:
        r["scientific_name"] = sci
        r["common_name"] = common
    all_records.extend(species_recs)
    time.sleep(1.0)

df = pd.DataFrame(all_records)
print(f"\n  total records pulled: {len(df)}")
print(f"  cols: {list(df.columns)[:20]}")

OUT_META = ROOT / "range_data" / "xc_metadata_4species_conus.parquet"
OUT_META.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_META, index=False)
print(f"  saved: {OUT_META}")

# ---------- 2. Filter to usable recordings ----------
print(f"\n[2] Filtering to song-type + has-coords + quality ...")
df["lat_num"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon_num"] = pd.to_numeric(df["lon"], errors="coerce")
df["q_score"] = df["q"].fillna("no score")
df["type_norm"] = df["type"].astype(str).str.lower()

n0 = len(df)
df_song = df[df["type_norm"].str.contains("song", na=False, regex=False)].copy()
print(f"  after type contains 'song': {len(df_song)} (was {n0})")
df_song = df_song.dropna(subset=["lat_num","lon_num"])
print(f"  after has coords: {len(df_song)}")
# Filter out extremely low-quality (E = "no score"-equivalent, very poor); keep A/B/C/no score
df_song = df_song[~df_song["q_score"].isin(["E","e","no score"])].copy()
print(f"  after quality A/B/C (drop D, E, no score): {len(df_song)}")

print(f"\n  by species:")
print(df_song.groupby("common_name").size().to_string())

# ---------- 3. Fetch BCR shapefile (USGS ScienceBase) ----------
print(f"\n[3] Fetching USFWS Bird Conservation Region (BCR) shapefile ...")
BCR_DIR = ROOT / "range_data" / "bcr_shapefile"
BCR_DIR.mkdir(parents=True, exist_ok=True)
# USFWS BCR shapefile from NABCI (North American Bird Conservation Initiative)
BCR_URL = "https://nabci-us.org/wp-content/uploads/2010/07/BCR.zip"
local_zip = BCR_DIR / "BCR.zip"
if not local_zip.exists():
    print(f"  downloading from {BCR_URL} ...")
    try:
        req = urllib.request.Request(BCR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            local_zip.write_bytes(r.read())
        print(f"  saved: {local_zip} ({local_zip.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"  ERROR fetching BCR shapefile: {e}")
        print(f"  trying alternative URL ...")
        # USGS ScienceBase alternative
        ALT = "https://www.sciencebase.gov/catalog/file/get/4fb697b2e4b03ad19d64b47f"
        try:
            req = urllib.request.Request(ALT, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                local_zip.write_bytes(r.read())
            print(f"  saved (alt): {local_zip}")
        except Exception as e2:
            print(f"  alt failed too: {e2}")
            print(f"  >>> SKIPPING spatial join; will report point counts only")
            BCR_URL = None
else:
    print(f"  already downloaded: {local_zip}")

# ---------- 4. Spatial join ----------
print(f"\n[4] Spatial join: recordings -> BCR ...")
try:
    import geopandas as gpd
    from shapely.geometry import Point
    # Extract shapefile
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(BCR_DIR / "extracted")
    shp_files = list((BCR_DIR / "extracted").rglob("*.shp"))
    print(f"  shp files found: {[s.name for s in shp_files]}")
    if not shp_files:
        print(f"  no shapefile inside zip; cannot spatial join")
        raise FileNotFoundError("no .shp")
    bcr = gpd.read_file(shp_files[0])
    print(f"  BCR polygons: {len(bcr)}  cols: {list(bcr.columns)[:10]}")
    bcr = bcr.to_crs("EPSG:5070")
    pts = gpd.GeoDataFrame(
        df_song[["common_name","scientific_name","ssp","lat_num","lon_num","id","q_score","length"]].copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df_song["lon_num"], df_song["lat_num"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    joined = gpd.sjoin(pts, bcr, how="left", predicate="within")
    n_matched = joined["index_right"].notna().sum()
    print(f"  spatial-joined: {n_matched} / {len(pts)} matched to a BCR")

    # Identify BCR id column
    candidate_cols = [c for c in bcr.columns if "BCR" in c.upper() or "REGION" in c.upper()]
    print(f"  BCR id candidate cols: {candidate_cols}")
    bcr_col = candidate_cols[0] if candidate_cols else "BCRNumber"
    name_col = next((c for c in bcr.columns if c.upper() in ["BCRNAME","NAME","BCR_NAME","REGION_NAM"]), None)
    print(f"  using BCR col: {bcr_col}  name col: {name_col}")

    # ---------- 5. Cell-count matrix ----------
    print(f"\n[5] Per-(species, BCR) cell counts:")
    joined["bcr_id"] = joined[bcr_col]
    cell_counts = joined.dropna(subset=["bcr_id"]).groupby(["common_name","bcr_id"]).size().unstack(fill_value=0)
    print(cell_counts.to_string())

    print(f"\n  cells with N >= 10 (per species):")
    print((cell_counts >= 10).sum(axis=1).to_string())
    print(f"\n  cells with N >= 20 (per species):")
    print((cell_counts >= 20).sum(axis=1).to_string())

    # Save
    cell_counts.to_csv(ROOT / "range_data" / "cell_counts_species_bcr.csv")
    joined.drop(columns="geometry").to_parquet(ROOT / "range_data" / "xc_metadata_joined_bcr.parquet", index=False)
    print(f"\n  saved cell_counts.csv + xc_metadata_joined_bcr.parquet")

except Exception as e:
    print(f"  spatial join failed: {type(e).__name__}: {e}")
    print(f"\n[FALLBACK] reporting by state instead of BCR:")
    by_state = df_song.groupby(["common_name","cnty"]).size().unstack(fill_value=0)
    print(by_state.head(30).to_string())

print(f"\n=== Cell-availability scope DONE ===")
