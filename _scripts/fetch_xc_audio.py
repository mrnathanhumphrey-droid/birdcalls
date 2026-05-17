"""Phase 1, step 1: fetch xeno-canto audio files for the 1,344 cell-eligible
recordings (after BCR join). Resumable, throttled, per-file logged.

Input:  range_data/xc_metadata_joined_bcr.parquet  (1,344 records)
Output: recordings/xeno_canto/XC{id}.mp3  +  recordings/xeno_canto/_manifest.csv

Resume: if XC{id}.mp3 already exists with non-zero size, skip.
Throttle: 1.5 seconds between requests (xeno-canto courtesy limit).
"""
import os, sys, urllib.request, urllib.error, pathlib, time, json
import pandas as pd

ROOT = pathlib.Path(r"D:/Bird Song")
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
OUT  = ROOT / "recordings" / "xeno_canto"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "_manifest.csv"

# Load API key
for line in open(ROOT / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v
KEY = os.environ["XENOCANTO_API_KEY"]

THROTTLE_SEC = 1.5
TIMEOUT = 120
USER_AGENT = "BirdCallsResearch/1.0 (https://github.com/mrnathanhumphrey-droid/birdcalls)"

# Need the full xc_metadata (joined parquet doesn't have the file URL).
# Re-merge from the raw parquet to get the audio URL field.
joined = pd.read_parquet(META)
raw    = pd.read_parquet(ROOT / "range_data" / "xc_metadata_4species_conus.parquet")
joined = joined.merge(raw[["id","file","file-name"]], on="id", how="left", suffixes=("","_raw"))
joined["id"] = joined["id"].astype(str)
print(f"records to fetch: {len(joined)}")
print(f"   col sample: {[c for c in joined.columns if c in ('id','file','file-name','common_name','bcr','length')]}")
missing_url = joined["file"].isna() | (joined["file"] == "")
print(f"   missing file URL: {missing_url.sum()}")
joined = joined[~missing_url].copy()
print(f"   fetchable: {len(joined)}")

# Load existing manifest if any
if MANIFEST.exists():
    mani = pd.read_csv(MANIFEST, dtype={"id":str})
    done_ids = set(mani[mani["status"]=="ok"]["id"])
    print(f"   already in manifest as OK: {len(done_ids)}")
else:
    mani = pd.DataFrame(columns=["id","common_name","bcr","status","path","size","error","wall_sec"])
    done_ids = set()

# Also skip any file that already exists on disk (any audio extension)
on_disk_ids = set()
for ext in ("mp3","wav","flac","ogg","m4a"):
    on_disk_ids |= {p.stem.replace("XC","") for p in OUT.glob(f"XC*.{ext}") if p.stat().st_size > 0}
done_ids |= on_disk_ids
print(f"   on-disk OR manifest-OK: {len(done_ids)}")

to_fetch = joined[~joined["id"].isin(done_ids)].copy()
print(f"\nTO FETCH THIS RUN: {len(to_fetch)}")
print(f"Throttle: {THROTTLE_SEC}s between requests. ETA: {len(to_fetch) * (THROTTLE_SEC + 3) / 60:.1f} min\n")

if len(to_fetch) == 0:
    print("All files already fetched. Exiting.")
    sys.exit(0)

new_rows = []
t_start = time.time()
errors = 0
ok = 0

for i, r in enumerate(to_fetch.itertuples(index=False), start=1):
    xc_id = r.id
    url = r.file
    if not url.startswith("http"):
        url = "https://" + url.lstrip("/")
    # xeno-canto v3 audio: may require ?key=KEY param
    sep = "&" if "?" in url else "?"
    full_url = f"{url}{sep}key={KEY}"

    t0 = time.time()
    out_path = None
    tmp_path = OUT / f"XC{xc_id}.tmp"
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            ct = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read()
        if "wav" in ct:    ext = "wav"
        elif "mpeg" in ct or "mp3" in ct: ext = "mp3"
        elif "flac" in ct: ext = "flac"
        elif "ogg" in ct:  ext = "ogg"
        else:              ext = "mp3"  # default
        out_path = OUT / f"XC{xc_id}.{ext}"
        tmp_path.write_bytes(data)
        size = tmp_path.stat().st_size
        if size < 1024:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"file too small: {size} bytes")
        tmp_path.rename(out_path)
        status = "ok"
        err = ""
        ok += 1
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, TimeoutError) as e:
        status = "error"
        size = 0
        err = f"{type(e).__name__}: {e}"
        errors += 1
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    wall = time.time() - t0
    new_rows.append({
        "id": xc_id, "common_name": r.common_name, "bcr": r.bcr,
        "status": status, "path": str(out_path) if status == "ok" else "",
        "size": size, "error": err, "wall_sec": wall,
    })

    # Progress every 25 files
    if i % 25 == 0 or i == len(to_fetch):
        elapsed = time.time() - t_start
        rate = i / elapsed
        eta = (len(to_fetch) - i) / rate if rate > 0 else 0
        print(f"[{i:>5}/{len(to_fetch)}]  ok={ok}  err={errors}  "
              f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
              f"last_ok_size={size//1024 if status=='ok' else 0}KB")

    # Persist manifest every 50 files in case of crash
    if i % 50 == 0:
        mani_new = pd.concat([mani, pd.DataFrame(new_rows)], ignore_index=True)
        mani_new.to_csv(MANIFEST, index=False)

    time.sleep(THROTTLE_SEC)

# Final manifest write
mani_new = pd.concat([mani, pd.DataFrame(new_rows)], ignore_index=True)
mani_new.to_csv(MANIFEST, index=False)
elapsed = time.time() - t_start
print(f"\n=== DONE ===")
print(f"  ok: {ok}  err: {errors}  total: {len(to_fetch)}")
print(f"  wall: {elapsed/60:.1f} min")
total_size_mb = sum(p.stat().st_size for p in OUT.glob("XC*.*") if p.suffix.lower() in (".mp3",".wav",".flac",".ogg",".m4a")) / 1024 / 1024
print(f"  total audio on disk: {total_size_mb:.0f} MB ({total_size_mb/1024:.2f} GB)")
print(f"  manifest: {MANIFEST}")
