"""Cross-species 3D UMAP for the bird-song 3D scatter visualization.

The existing dialect-clustering pipeline runs UMAP WITHIN each species
(per-species 8-dim reduction → HDBSCAN). For the 3D scatter we want a
single global coordinate frame so all 1344 recordings can be compared
in one Three.js view.

Output:
  acoustic_features/umap_3d_v1.parquet — per-recording 3D coords +
    species + dialect_cluster + BCR + recordist for the front-end
  viz/birdsong_3d_points.json — same data trimmed to what the viz needs
"""
import pathlib, sys, io, json
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass
import numpy as np
import pandas as pd
import umap

ROOT = pathlib.Path(r"D:/Bird Song")
EMB = ROOT / "acoustic_features" / "birdnet_embeddings_v1.parquet"
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
DIALECT = ROOT / "acoustic_features" / "dialect_clusters_v1.parquet"
OUT_PARQ = ROOT / "acoustic_features" / "umap_3d_v1.parquet"
OUT_JSON = ROOT / "viz" / "birdsong_3d_points.json"


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== Cross-species 3D UMAP ===")

    emb = pd.read_parquet(EMB)
    emb = emb[emb["ok"]].copy()
    emb["id"] = emb["id"].astype(str)
    print(f"  embeddings ok: {len(emb)}")

    meta = pd.read_parquet(META)
    meta["id"] = meta["id"].astype(str)
    dialect = pd.read_parquet(DIALECT)
    dialect["id"] = dialect["id"].astype(str)

    meta_cols = [c for c in ["id", "common_name", "scientific_name", "ssp",
                              "bcr", "BCR_NAME", "lat_num", "lon_num", "q_score"]
                 if c in meta.columns]
    df = emb.merge(meta[meta_cols], on="id", how="inner")
    df = df.merge(dialect[["id", "dialect_cluster"]], on="id", how="left")
    df["dialect_cluster"] = df["dialect_cluster"].fillna(-1).astype(int)
    print(f"  joined: {len(df)}")

    emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
    assert len(emb_cols) == 1024
    X = df[emb_cols].values

    print(f"  fitting global UMAP n_components=3, n_neighbors=30, metric=cosine ...")
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=20260518)
    X3 = reducer.fit_transform(X)
    print(f"  X3 shape: {X3.shape}; ranges: "
          f"x=[{X3[:,0].min():.2f},{X3[:,0].max():.2f}], "
          f"y=[{X3[:,1].min():.2f},{X3[:,1].max():.2f}], "
          f"z=[{X3[:,2].min():.2f},{X3[:,2].max():.2f}]")

    keep_cols = [c for c in ["id", "common_name", "scientific_name", "ssp",
                              "bcr", "BCR_NAME", "lat_num", "lon_num", "q_score",
                              "dialect_cluster"]
                 if c in df.columns]
    df_out = df[keep_cols].copy()
    df_out["umap3_x"] = X3[:, 0]
    df_out["umap3_y"] = X3[:, 1]
    df_out["umap3_z"] = X3[:, 2]
    df_out.to_parquet(OUT_PARQ, index=False)
    print(f"  -> {OUT_PARQ.relative_to(ROOT)}")

    # Per-recording dialect-cluster label needs species prefix for distinct color
    # (each species has its own [-1, 0, 1, 2, ...] labels)
    df_out["species_dialect"] = df_out.apply(
        lambda r: f"{r['common_name']}/c{r['dialect_cluster']}" if r['dialect_cluster'] >= 0
                  else f"{r['common_name']}/noise", axis=1)

    payload = {
        "n_points": len(df_out),
        "umap_params": {"n_components": 3, "n_neighbors": 30, "min_dist": 0.1,
                        "metric": "cosine", "random_state": 20260518},
        "fit_on": "BirdNET embedding-mean (1024-dim) over all 1344 recordings, 4 pilot species",
        "species_list": sorted(df_out["common_name"].unique().tolist()),
        "n_species_dialects": int(df_out["species_dialect"].nunique()),
        "points": [
            {
                "id": str(row["id"]),
                "x": float(row["umap3_x"]),
                "y": float(row["umap3_y"]),
                "z": float(row["umap3_z"]),
                "species": row["common_name"],
                "dialect": int(row["dialect_cluster"]),
                "species_dialect": row["species_dialect"],
                "bcr_name": row["BCR_NAME"] if pd.notna(row["BCR_NAME"]) else "unknown",
                "ssp": str(row.get("ssp", "")) if pd.notna(row.get("ssp")) else "",
                "lat": float(row["lat_num"]) if "lat_num" in row.index and pd.notna(row["lat_num"]) else None,
                "lon": float(row["lon_num"]) if "lon_num" in row.index and pd.notna(row["lon_num"]) else None,
            }
            for _, row in df_out.iterrows()
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"  -> {OUT_JSON.relative_to(ROOT)}  ({OUT_JSON.stat().st_size:,} bytes)")

    # Print cluster summary
    print(f"\n=== Cluster summary ===")
    summ = (df_out.groupby(["common_name", "dialect_cluster"]).size()
                  .rename("n").reset_index())
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
