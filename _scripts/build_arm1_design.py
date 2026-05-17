"""Arm 1 design build: per-recording spectral + structural feature distances
within each (species × BCR × dialect_cluster) cell.

Output rows: one per recording, columns:
  id, common_name, BCR, dialect_cluster, cell_id,
  d_spectral (cosine dist from species centroid in BirdNET-emb space),
  d_structural (proxy: composite z-score across SoundPlot structural feats),
  log_n_cell (for offset)

Input:  acoustic_features/birdnet_embeddings_v1.parquet
        acoustic_features/soundplot_features_v1.parquet
        acoustic_features/dialect_clusters_v1.parquet
        range_data/xc_metadata_joined_bcr.parquet
Output: analysis/arm1_design.parquet
"""
import pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

ROOT = pathlib.Path(r"D:/Bird Song")

print("[1] Loading inputs ...")
emb    = pd.read_parquet(ROOT / "acoustic_features" / "birdnet_embeddings_v1.parquet")
struct = pd.read_parquet(ROOT / "acoustic_features" / "structural_features_v1.parquet")
dc     = pd.read_parquet(ROOT / "acoustic_features" / "dialect_clusters_v1.parquet")
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
# Replace NaN structural features with zeros (recordings with no syllables detected)
for c in ["n_syllables","syllable_inventory_size","transition_entropy","lz_complexity","lz_ratio"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
print(f"  rows joined: {len(df)}")

emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
# Spectral distance: Euclidean distance from species-centroid BirdNET embedding.
# Cosine fails here because BirdNET embeddings are highly aligned within species
# (the model was trained for species ID), so cosine distances all collapse to ~0.
# Euclidean distance captures the magnitude of within-species deviation.
print("\n[2] Computing spectral distance (Euclidean from species centroid) ...")
X = df[emb_cols].values  # (N, 1024)
sp_codes = pd.Categorical(df["common_name"]).codes
sp_labels = list(pd.Categorical(df["common_name"]).categories)
centroids = np.zeros((len(sp_labels), X.shape[1]))
for k in range(len(sp_labels)):
    centroids[k] = X[sp_codes == k].mean(axis=0)
C = centroids[sp_codes]  # (N, 1024)
diff = X - C
d_eucl = np.sqrt((diff * diff).sum(axis=1))
df["d_spectral"] = d_eucl
print(f"  d_spectral stats:"); print(df["d_spectral"].describe())
print(f"  per-species mean d_spectral:")
for k, sp in enumerate(sp_labels):
    print(f"    {sp:<24}  mean={d_eucl[sp_codes==k].mean():.3f}  std={d_eucl[sp_codes==k].std():.3f}")

# Structural distance: Euclidean distance from species centroid in the 5-d
# structural feature space (n_syllables, inventory_size, transition_entropy,
# lz_complexity, lz_ratio), each z-scored within species first.
print(f"\n[3] Structural distance (z-scored Euclidean from species centroid) ...")
struct_cols = ["n_syllables","syllable_inventory_size","transition_entropy","lz_complexity","lz_ratio"]
S_arr = df[struct_cols].values  # (N, 5)
sp_codes = pd.Categorical(df["common_name"]).codes
sp_labels = list(pd.Categorical(df["common_name"]).categories)
# Z-score within species per column
S_z = np.zeros_like(S_arr, dtype=float)
for k in range(len(sp_labels)):
    m = sp_codes == k
    block = S_arr[m]
    mu = block.mean(axis=0)
    sd = block.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    S_z[m] = (block - mu) / sd_safe
# After z-scoring within species, species centroid = 0 vector
# Distance is just the L2 norm of the z-scored row
df["d_structural"] = np.sqrt((S_z * S_z).sum(axis=1))
print(f"  d_structural stats:"); print(df["d_structural"].describe())
print(f"  per-species mean d_structural:")
for k, sp in enumerate(sp_labels):
    print(f"    {sp:<24}  mean={df.loc[sp_codes==k, 'd_structural'].mean():.3f}  std={df.loc[sp_codes==k, 'd_structural'].std():.3f}")

# Build cell_id (species_BCR_dialectcluster)
df["bcr_str"] = df["bcr"].astype(str)
df["dc_str"]  = df["dialect_cluster"].astype(str)
df["cell_id"] = df["common_name"] + "::" + df["bcr_str"] + "::" + df["dc_str"]
# Filter to non-noise (dialect_cluster >= 0)
n0 = len(df)
df_cell = df[df["dialect_cluster"] >= 0].copy()
print(f"\n[4] Cell-eligible (non-noise): {len(df_cell)} / {n0}")

# n per cell
cell_n = df_cell.groupby("cell_id").size().rename("n_cell")
df_cell = df_cell.merge(cell_n, left_on="cell_id", right_index=True)
df_cell["log_n_cell"] = np.log(df_cell["n_cell"])
print(f"  cells: {df_cell['cell_id'].nunique()}")
print(f"  cell n distribution: min={df_cell['n_cell'].min()} median={df_cell['n_cell'].median()} max={df_cell['n_cell'].max()}")
print(f"  n>=10 cells: {(cell_n>=10).sum()}  n>=5 cells: {(cell_n>=5).sum()}")

OUT = ROOT / "analysis" / "arm1_design.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)
df_cell.to_parquet(OUT, index=False)
print(f"\nSaved {OUT}")
