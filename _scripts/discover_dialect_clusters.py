"""Within-species HDBSCAN dialect-cluster discovery on BirdNET embeddings.

For each pilot species: UMAP-reduce 1024-dim BirdNET embedding-mean to
8-dim, then HDBSCAN cluster. Each recording gets a dialect_cluster id
(or -1 for noise). Saved as per-recording table for downstream cell join.

Input:  acoustic_features/birdnet_embeddings_v1.parquet
        range_data/xc_metadata_joined_bcr.parquet
Output: acoustic_features/dialect_clusters_v1.parquet
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import umap, hdbscan

ROOT = pathlib.Path(r"D:/Bird Song")
EMB = ROOT / "acoustic_features" / "birdnet_embeddings_v1.parquet"
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
OUT = ROOT / "acoustic_features" / "dialect_clusters_v1.parquet"

print("[1] Loading embeddings + metadata ...")
emb = pd.read_parquet(EMB)
emb = emb[emb["ok"]].copy()
print(f"  embeddings ok: {len(emb)}")
meta = pd.read_parquet(META)
meta["id"] = meta["id"].astype(str)
emb["id"] = emb["id"].astype(str)
df = emb.merge(meta[["id","common_name","scientific_name","ssp","bcr","BCR_NAME"]],
               on="id", how="inner")
print(f"  joined: {len(df)}")

# Pull embedding-mean columns only (1024)
emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
print(f"  embedding-mean cols: {len(emb_cols)}")
assert len(emb_cols) == 1024, f"expected 1024, got {len(emb_cols)}"

cluster_rows = []
for sp, sub in df.groupby("common_name"):
    n = len(sub)
    print(f"\n[2] {sp}: n={n} recordings")
    X = sub[emb_cols].values
    # UMAP -> 8-dim, then HDBSCAN
    rs = 20260516
    reducer = umap.UMAP(n_components=8, n_neighbors=15, min_dist=0.0,
                        metric="cosine", random_state=rs)
    Xr = reducer.fit_transform(X)
    # HDBSCAN: min_cluster_size as 5% of species n (min 5), min_samples 3
    mcs = max(5, int(n * 0.05))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=3,
                                cluster_selection_method="eom")
    labels = clusterer.fit_predict(Xr)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  UMAP->8d + HDBSCAN(mcs={mcs}, ms=3): {n_clusters} clusters, "
          f"{n_noise} noise ({100*n_noise/n:.1f}%)")
    for i, (xc_id, lab) in enumerate(zip(sub["id"].values, labels)):
        cluster_rows.append({
            "id": xc_id,
            "common_name": sp,
            "dialect_cluster": int(lab),  # -1 = noise
            "umap_x": float(Xr[i, 0]),
            "umap_y": float(Xr[i, 1]),
        })

out_df = pd.DataFrame(cluster_rows)
print(f"\n[3] Saving {len(out_df)} records to {OUT}")
out_df.to_parquet(OUT, index=False)
print("\n=== DONE ===")
print(out_df.groupby(["common_name","dialect_cluster"]).size().head(40).to_string())
