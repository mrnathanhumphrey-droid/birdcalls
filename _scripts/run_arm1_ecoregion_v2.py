"""Phase 2 robustness axis 2: geographic refinement — swap BCR for L3 ecoregion.

Rebuilds Arm 1 design with L3 ecoregion (EPA, 105 NA ecoregions, finer than 36 BCRs).
Re-fits the v0.2 Arm 1 model. If spectral PASS holds at L3 → geography-axis robust.
"""
import os
RTOOLS = "C:/Users/Nate/.cmdstan/RTools40"
os.environ["PATH"] = f"{RTOOLS}/mingw64/bin;{RTOOLS}/usr/bin;" + os.environ.get("PATH", "")
import pathlib, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from cmdstanpy import CmdStanModel

ROOT = pathlib.Path(r"D:/Bird Song")
DESIGN = ROOT / "analysis" / "arm1_design_v2.parquet"
ECO    = ROOT / "range_data" / "ecoregion_l3" / "extracted" / "us_eco_l3.shp"
STAN   = ROOT / "analysis" / "fit_arm1.stan"
OUT    = ROOT / "analysis" / "arm1_ecoregion_v2"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading existing Arm 1 design + L3 ecoregion shapefile ...")
df = pd.read_parquet(DESIGN)
print(f"  design rows: {len(df)}")
eco = gpd.read_file(ECO)
print(f"  L3 ecoregions: {len(eco)}, cols: {list(eco.columns)[:8]}")
eco = eco.to_crs("EPSG:5070")
# Find name/id col
id_col = next((c for c in eco.columns if c.upper() in ("US_L3CODE","L3_KEY","ECOREGION","L3_CODE","NA_L3CODE")), None)
print(f"  using ecoregion id col: {id_col}")

print("\n[2] Spatial-join recordings to L3 ecoregion ...")
# Need lat/lon from joined parquet
meta = pd.read_parquet(ROOT / "range_data" / "xc_metadata_joined_bcr.parquet")
meta["id"] = meta["id"].astype(str)
df["id"] = df["id"].astype(str)
df = df.merge(meta[["id","lat_num","lon_num"]], on="id", how="left")
pts = gpd.GeoDataFrame(
    df[["id","lat_num","lon_num"]],
    geometry=[Point(lon, lat) for lon, lat in zip(df["lon_num"], df["lat_num"])],
    crs="EPSG:4326",
).to_crs("EPSG:5070")
joined = gpd.sjoin(pts, eco[[id_col, "geometry"]], how="left", predicate="within")
print(f"  matched to ecoregion: {joined[id_col].notna().sum()} / {len(joined)}")

df["eco"] = joined[id_col].astype(str).values
df = df.dropna(subset=["eco"]).copy()
df = df[df["eco"] != "nan"].copy()
print(f"  after dropping unmatched: {len(df)}")

# Redefine cell at species × ecoregion × dialect_cluster
df["cell_id_eco"] = df["common_name"] + "::" + df["eco"] + "::" + df["dialect_cluster"].astype(str)
cell_n = df.groupby("cell_id_eco").size()
df = df[df["cell_id_eco"].isin(cell_n[cell_n >= 2].index)].copy()
print(f"  rows in non-singleton ecoregion cells: {len(df)}")
print(f"  unique ecoregions touched: {df['eco'].nunique()}")
print(f"  ecoregion cells: {df['cell_id_eco'].nunique()}")
print(f"  cell n: min={df.groupby('cell_id_eco').size().min()} "
      f"median={df.groupby('cell_id_eco').size().median():.0f} "
      f"max={df.groupby('cell_id_eco').size().max()}")

print("\n[3] Fitting ...")
model = CmdStanModel(stan_file=str(STAN))

results = {}
sp_codes  = pd.Categorical(df["common_name"]).codes + 1
eco_codes = pd.Categorical(df["eco"]).codes + 1
cell_codes= pd.Categorical(df["cell_id_eco"]).codes + 1
S = int(sp_codes.max()); B = int(eco_codes.max()); C = int(cell_codes.max())

for outcome in ["d_spectral", "d_structural"]:
    print(f"\n  Outcome: {outcome}")
    y_raw = df[outcome].values
    y_z = np.zeros_like(y_raw, dtype=float)
    for s_code in range(1, S+1):
        m = (sp_codes == s_code); v = y_raw[m]
        if np.std(v) > 1e-8: y_z[m] = (v - v.mean()) / v.std()
    t0 = time.time()
    fit = model.sample(
        data={
            "N": len(df), "S": S, "B": B, "C": C,
            "species": sp_codes.tolist(),
            "bcr": eco_codes.tolist(),  # baseline grouping = L3 ecoregion
            "cell": cell_codes.tolist(),
            "y": y_z.tolist(),
        },
        chains=4, parallel_chains=4,
        iter_warmup=1000, iter_sampling=1000,
        adapt_delta=0.95, max_treedepth=10,
        show_progress=False, seed=20260517,
    )
    print(f"    sampled in {time.time()-t0:.1f}s")
    post = fit.draws_pd()
    fc = post["frac_cell"]
    rec = {
        "outcome": outcome, "n": int(len(df)), "S": S, "ecoregions": B, "C": C,
        "frac_cell_mean": float(fc.mean()),
        "frac_cell_q025": float(fc.quantile(0.025)),
        "frac_cell_q975": float(fc.quantile(0.975)),
        "passes_5pct": bool(fc.quantile(0.025) > 0.05),
        "ci_clean_pos": bool(fc.quantile(0.025) > 0),
    }
    print(f"    frac_cell mean={rec['frac_cell_mean']:.3f} "
          f"95% CrI [{rec['frac_cell_q025']:.3f}, {rec['frac_cell_q975']:.3f}]  "
          f"passes 5%: {rec['passes_5pct']}")
    results[outcome] = rec

with open(OUT / "ecoregion_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {OUT}/ecoregion_results.json")
