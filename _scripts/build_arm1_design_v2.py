"""Arm 1 design build — v2 on REAL BirdNET embeddings + v2 dialect clusters.

Same procedure as build_arm1_design.py but reads:
  - birdnet_embeddings_v2.parquet  (real 1024-dim)
  - dialect_clusters_v2.parquet    (from real embeddings)
"""
import pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:/Bird Song")

print("[1] Loading v2 inputs ...")
emb    = pd.read_parquet(ROOT / "acoustic_features" / "birdnet_embeddings_v2.parquet")
struct = pd.read_parquet(ROOT / "acoustic_features" / "structural_features_v1.parquet")
dc     = pd.read_parquet(ROOT / "acoustic_features" / "dialect_clusters_v2.parquet")
meta   = pd.read_parquet(ROOT / "range_data" / "xc_metadata_joined_bcr.parquet")
for d in (emb, struct, dc, meta):
    d["id"] = d["id"].astype(str)

emb_ok = emb[emb["ok"]]
df = (dc.merge(emb_ok[["id"] + [c for c in emb_ok.columns if c.startswith("emb_mean_")]],
               on="id", how="inner")
        .merge(struct[["id","n_syllables","syllable_inventory_size",
                       "transition_entropy","lz_complexity","lz_ratio"]],
               on="id", how="left")
        .merge(meta[["id","bcr","BCR_NAME","scientific_name","ssp"]], on="id", how="inner"))
for c in ["n_syllables","syllable_inventory_size","transition_entropy","lz_complexity","lz_ratio"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
print(f"  rows joined: {len(df)}")

emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
print(f"\n[2] Computing spectral distance (Euclidean from species centroid, on REAL embeddings) ...")
X = df[emb_cols].values
sp_codes = pd.Categorical(df["common_name"]).codes
sp_labels = list(pd.Categorical(df["common_name"]).categories)
centroids = np.zeros((len(sp_labels), X.shape[1]))
for k in range(len(sp_labels)):
    centroids[k] = X[sp_codes == k].mean(axis=0)
C = centroids[sp_codes]
diff = X - C
d_eucl = np.sqrt((diff * diff).sum(axis=1))
df["d_spectral"] = d_eucl
print(f"  d_spectral: mean={d_eucl.mean():.3f}  std={d_eucl.std():.3f}")
print(f"  per-species:")
for k, sp in enumerate(sp_labels):
    print(f"    {sp:<24}  mean={d_eucl[sp_codes==k].mean():.3f}  std={d_eucl[sp_codes==k].std():.3f}")

print(f"\n[3] Structural distance (z-scored Euclidean from species centroid) ...")
struct_cols = ["n_syllables","syllable_inventory_size","transition_entropy","lz_complexity","lz_ratio"]
S_arr = df[struct_cols].values
S_z = np.zeros_like(S_arr, dtype=float)
for k in range(len(sp_labels)):
    m = sp_codes == k
    block = S_arr[m]
    mu = block.mean(axis=0); sd = block.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    S_z[m] = (block - mu) / sd_safe
df["d_structural"] = np.sqrt((S_z * S_z).sum(axis=1))

df["bcr_str"] = df["bcr"].astype(str)
df["dc_str"]  = df["dialect_cluster"].astype(str)
df["cell_id"] = df["common_name"] + "::" + df["bcr_str"] + "::" + df["dc_str"]
n0 = len(df)
df_cell = df[df["dialect_cluster"] >= 0].copy()
print(f"\n[4] Cell-eligible (non-noise): {len(df_cell)} / {n0}")

cell_n = df_cell.groupby("cell_id").size().rename("n_cell")
df_cell = df_cell.merge(cell_n, left_on="cell_id", right_index=True)
df_cell["log_n_cell"] = np.log(df_cell["n_cell"])
print(f"  cells: {df_cell['cell_id'].nunique()}")
print(f"  cell n: min={df_cell['n_cell'].min()} median={df_cell['n_cell'].median():.0f} max={df_cell['n_cell'].max()}")

OUT = ROOT / "analysis" / "arm1_design_v2.parquet"
df_cell.to_parquet(OUT, index=False)
print(f"\nSaved {OUT}")
