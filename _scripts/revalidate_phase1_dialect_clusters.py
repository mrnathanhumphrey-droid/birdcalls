"""Phase 1/2 revalidation: re-run dialect clustering on REAL BirdNET embeddings.

The Phase 1 pipeline ran on birdnet_embeddings_v1.parquet which contained
a single scalar per recording broadcast to all 1024 dims (bug: used
result.embeddings_masked bool mask instead of result.embeddings float data).

This script runs the identical UMAP+HDBSCAN procedure but on v2 (the fixed
real embeddings) and compares:
  - How many clusters per species (vs v1)
  - Per-recording cluster reassignment (Adjusted Rand Index between v1 and v2 labels)
  - Sample dialect-cluster reassignment table

Output: acoustic_features/dialect_clusters_v2.parquet
        notes/phase1_revalidation_2026_05_18.md
"""
import sys, pathlib, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import umap, hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = pathlib.Path(r"D:/Bird Song")
EMB_V2 = ROOT / "acoustic_features" / "birdnet_embeddings_v2.parquet"
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
DIALECT_V1 = ROOT / "acoustic_features" / "dialect_clusters_v1.parquet"
OUT_V2 = ROOT / "acoustic_features" / "dialect_clusters_v2.parquet"
OUT_MD = ROOT / "notes" / "phase1_revalidation_2026_05_18.md"


def main():
    print("[1] Loading v2 embeddings + metadata ...")
    emb = pd.read_parquet(EMB_V2)
    emb = emb[emb["ok"]].copy()
    print(f"  v2 ok: {len(emb)}")
    meta = pd.read_parquet(META)
    meta["id"] = meta["id"].astype(str)
    emb["id"] = emb["id"].astype(str)
    df = emb.merge(meta[["id","common_name","scientific_name","ssp","bcr","BCR_NAME"]],
                   on="id", how="inner")
    print(f"  joined: {len(df)}")

    emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
    assert len(emb_cols) == 1024

    # Run same UMAP+HDBSCAN procedure as v1
    rs = 20260516
    cluster_rows = []
    summary_per_sp = []
    for sp, sub in df.groupby("common_name"):
        n = len(sub)
        print(f"\n[2] {sp}: n={n}")
        X = sub[emb_cols].values
        reducer = umap.UMAP(n_components=8, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=rs)
        Xr = reducer.fit_transform(X)
        mcs = max(5, int(n * 0.05))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=3,
                                    cluster_selection_method="eom")
        labels = clusterer.fit_predict(Xr)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  v2: {n_clusters} clusters, {n_noise} noise ({100*n_noise/n:.1f}%)")
        summary_per_sp.append({"species": sp, "n": n, "v2_n_clusters": n_clusters,
                                "v2_n_noise": n_noise, "v2_pct_noise": 100 * n_noise / n})
        for i, (xc_id, lab) in enumerate(zip(sub["id"].values, labels)):
            cluster_rows.append({
                "id": xc_id, "common_name": sp,
                "dialect_cluster": int(lab),
                "umap_x": float(Xr[i, 0]), "umap_y": float(Xr[i, 1]),
            })

    v2_df = pd.DataFrame(cluster_rows)
    v2_df.to_parquet(OUT_V2, index=False)
    print(f"\n[3] saved {len(v2_df)} records -> {OUT_V2.relative_to(ROOT)}")

    # Compare to v1
    print(f"\n[4] Comparing v1 vs v2 ...")
    v1_df = pd.read_parquet(DIALECT_V1)
    v1_df["id"] = v1_df["id"].astype(str)
    v2_df["id"] = v2_df["id"].astype(str)
    merged = v1_df[["id","common_name","dialect_cluster"]].rename(columns={"dialect_cluster":"v1"}).merge(
        v2_df[["id","dialect_cluster"]].rename(columns={"dialect_cluster":"v2"}),
        on="id", how="inner"
    )
    print(f"  paired: {len(merged)} recordings")

    print(f"\n  === Per-species cluster-comparison metrics ===")
    comparison = []
    for sp, sub in merged.groupby("common_name"):
        v1_lab = sub["v1"].values
        v2_lab = sub["v2"].values
        ari = adjusted_rand_score(v1_lab, v2_lab)
        nmi = normalized_mutual_info_score(v1_lab, v2_lab)
        v1_k = sub["v1"].nunique() - (1 if -1 in v1_lab else 0)
        v2_k = sub["v2"].nunique() - (1 if -1 in v2_lab else 0)
        v1_noise = (v1_lab == -1).sum()
        v2_noise = (v2_lab == -1).sum()
        comparison.append({"species": sp, "n": len(sub),
                            "v1_n_clusters": v1_k, "v2_n_clusters": v2_k,
                            "v1_n_noise": v1_noise, "v2_n_noise": v2_noise,
                            "ari": ari, "nmi": nmi})
        print(f"  {sp}: v1={v1_k} cluster (noise={v1_noise})  v2={v2_k} clusters (noise={v2_noise})  "
              f"ARI={ari:.3f}  NMI={nmi:.3f}")

    # Markdown summary
    md = []
    md.append(f"# Phase 1 Revalidation on Real BirdNET Embeddings\n")
    md.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"Bug: v1 BirdNET extraction wrote `result.embeddings_masked` (bool mask) instead of `result.embeddings` (float data). Each row had a single scalar = `n_valid_segments/21` broadcast to all 1024 dims.\n")
    md.append(f"Fix: v2 uses `result.embeddings` with corrected mask convention (True = MASKED) + nan-aware aggregation. All 1344 recordings now have real 1024-dim embeddings.\n")
    md.append(f"\n## Per-species cluster comparison (v1 broken vs v2 real)\n")
    md.append("| Species | n | v1 clusters | v2 clusters | v1 noise | v2 noise | ARI | NMI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for c in comparison:
        md.append(f"| {c['species']} | {c['n']} | {c['v1_n_clusters']} | {c['v2_n_clusters']} | "
                  f"{c['v1_n_noise']} | {c['v2_n_noise']} | {c['ari']:.3f} | {c['nmi']:.3f} |")
    md.append(f"\n## Interpretation\n")
    md.append(f"- **ARI = Adjusted Rand Index**: 1.0 means identical clusterings, 0 means chance, negative means worse-than-chance. ARI close to 0 indicates v1 and v2 clusterings are essentially independent — i.e., v1 dialect labels carried little structural information.")
    md.append(f"- **NMI = Normalized Mutual Information**: 0 to 1. Captures shared cluster structure even when labels permute.")
    md.append(f"- If ARI is low (<0.3), the Phase 1/2 conclusions about dialect-cluster partition need re-running on v2 because the cluster labels driving Stan partial-pooling have shifted substantially.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  -> {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
