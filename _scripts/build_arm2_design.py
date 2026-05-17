"""Arm 2 design: acoustic-niche partition cells.

Per-recording:
  - H3 hex level 5 (~252 km^2) from lat/lon  -> "community"
  - spectral_band (low/mid/high) from spectral_centroid_mean (SoundPlot)
  - time_of_day (dawn/day/dusk_night) from xc `time` field
  - cell_id = community + spectral_band + time_of_day

Filter to communities where >= 2 of our 4 species have recordings (sympatric).

Outcome: acoustic-niche distance from species centroid (z-Euclidean) in 2-d
  niche space (spectral_centroid_z, time_of_day_z).

Output: analysis/arm2_design.parquet
"""
import pathlib, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h3

ROOT = pathlib.Path(r"D:/Bird Song")

print("[1] Loading inputs ...")
sp_meta = pd.read_parquet(ROOT / "range_data" / "xc_metadata_4species_conus.parquet")
joined  = pd.read_parquet(ROOT / "range_data" / "xc_metadata_joined_bcr.parquet")
sound   = pd.read_parquet(ROOT / "acoustic_features" / "soundplot_features_v1.parquet")
struct  = pd.read_parquet(ROOT / "acoustic_features" / "structural_features_v1.parquet")
for d in (sp_meta, joined, sound, struct):
    d["id"] = d["id"].astype(str)

df = (joined[["id","common_name","lat_num","lon_num","bcr"]]
      .merge(sp_meta[["id","time"]], on="id", how="left")
      .merge(sound[["id","spectral_centroid_mean"]], on="id", how="left")
      .merge(struct[["id","n_syllables","syllable_inventory_size",
                     "transition_entropy","lz_complexity"]], on="id", how="left"))
df = df.dropna(subset=["lat_num","lon_num","spectral_centroid_mean"]).copy()
print(f"  rows: {len(df)}")

print("\n[2] H3 hex (level 5) community assignment ...")
df["h3"] = df.apply(lambda r: h3.latlng_to_cell(r["lat_num"], r["lon_num"], 5), axis=1)
hex_counts = df.groupby("h3")["common_name"].nunique()
sympatric_hexes = hex_counts[hex_counts >= 2].index
print(f"  total H3 cells: {df['h3'].nunique()}")
print(f"  sympatric (>=2 species): {len(sympatric_hexes)}")
df_sym = df[df["h3"].isin(sympatric_hexes)].copy()
print(f"  recordings in sympatric communities: {len(df_sym)}")

print("\n[3] Spectral band assignment (low / mid / high) ...")
def spec_band(c):
    if c < 2000: return "low"
    if c < 6000: return "mid"
    return "high"
df_sym["spec_band"] = df_sym["spectral_centroid_mean"].apply(spec_band)
print(df_sym["spec_band"].value_counts().to_string())

print("\n[4] Time-of-day bucket from xc `time` field ...")
def to_hour(t):
    if pd.isna(t) or not t: return None
    m = re.match(r"^(\d{1,2}):(\d{2})", str(t).strip())
    if m: return int(m.group(1))
    return None
df_sym["hour"] = df_sym["time"].apply(to_hour)
def time_bucket(h):
    if h is None: return "unknown"
    if 4 <= h < 8: return "dawn"
    if 8 <= h < 18: return "day"
    return "dusk_night"
df_sym["tod"] = df_sym["hour"].apply(time_bucket)
print(df_sym["tod"].value_counts().to_string())
# Drop unknown time-of-day (no signal for cell)
n0 = len(df_sym)
df_sym = df_sym[df_sym["tod"] != "unknown"].copy()
print(f"  after dropping unknown tod: {len(df_sym)} (was {n0})")

print("\n[5] Cell construction ...")
df_sym["cell_id"] = df_sym["h3"] + "::" + df_sym["spec_band"] + "::" + df_sym["tod"]
cell_n = df_sym.groupby("cell_id").size()
print(f"  cells: {df_sym['cell_id'].nunique()}")
print(f"  cell n distribution: min={cell_n.min()} median={cell_n.median()} max={cell_n.max()}")
print(f"  n>=2 cells: {(cell_n>=2).sum()}  n>=5 cells: {(cell_n>=5).sum()}")

# Drop singleton cells
keep = cell_n[cell_n >= 2].index
df_sym = df_sym[df_sym["cell_id"].isin(keep)].copy()
print(f"  rows after dropping singletons: {len(df_sym)}")

print("\n[6] Acoustic-niche distance from species centroid ...")
sp_codes = pd.Categorical(df_sym["common_name"]).codes
sp_labels = list(pd.Categorical(df_sym["common_name"]).categories)
# Spectral-niche outcome: z-scored spectral centroid within species
sc = df_sym["spectral_centroid_mean"].values
sc_z = np.zeros_like(sc, dtype=float)
for k in range(len(sp_labels)):
    m = sp_codes == k
    v = sc[m]
    if np.std(v) > 1e-8:
        sc_z[m] = (v - v.mean()) / v.std()
df_sym["d_spectral_niche"] = np.abs(sc_z)

# Structural-niche outcome: z-Euclidean on (n_syllables, inventory, entropy, lz_complexity)
ST = df_sym[["n_syllables","syllable_inventory_size","transition_entropy","lz_complexity"]].fillna(0).values
ST_z = np.zeros_like(ST, dtype=float)
for k in range(len(sp_labels)):
    m = sp_codes == k
    block = ST[m]
    mu = block.mean(axis=0); sd = block.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    ST_z[m] = (block - mu) / sd_safe
df_sym["d_structural_niche"] = np.sqrt((ST_z * ST_z).sum(axis=1))

print(f"  d_spectral_niche stats:"); print(df_sym["d_spectral_niche"].describe())
print(f"  d_structural_niche stats:"); print(df_sym["d_structural_niche"].describe())

OUT = ROOT / "analysis" / "arm2_design.parquet"
df_sym.to_parquet(OUT, index=False)
print(f"\nSaved {OUT}")
