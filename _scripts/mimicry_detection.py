"""Mimicry detection from BirdNET embeddings.

For each of 1344 recordings, compute cosine distance to all 4 species'
centroids in BirdNET-embedding space. Flag recordings where:

  distance(rec, labeled_species_centroid) > distance(rec, other_species_centroid)

i.e., the recording is closer to a DIFFERENT species' centroid than to its
own label. Candidate mimicry events.

Also report:
- "stretchers": recordings unusually FAR from their species centroid (could be
  outlier songs, unusual subspecies, or genuine cross-species drift)
- "tight singers": recordings unusually close — canonical representatives
"""
import pathlib, sys, io, json, time
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean

ROOT = pathlib.Path(r"D:/Bird Song")
EMB = ROOT / "acoustic_features" / "birdnet_embeddings_v2.parquet"
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
OUT_DIR = ROOT / "viz" / "mimicry"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"=== Mimicry detection from BirdNET embeddings ===")
    emb = pd.read_parquet(EMB)
    emb = emb[emb["ok"]].copy()
    emb["id"] = emb["id"].astype(str)
    meta = pd.read_parquet(META)
    meta["id"] = meta["id"].astype(str)
    df = emb.merge(meta[["id", "common_name", "BCR_NAME", "lat_num", "lon_num", "ssp"]],
                   on="id", how="inner")

    emb_cols = sorted([c for c in df.columns if c.startswith("emb_mean_")])
    print(f"  {len(df)} recordings; {len(emb_cols)} embedding dims")

    species_list = sorted(df["common_name"].unique().tolist())
    print(f"  species: {species_list}")

    X = df[emb_cols].values
    # Compute species centroids
    centroids = {}
    for sp in species_list:
        mask = df["common_name"] == sp
        centroids[sp] = X[mask].mean(axis=0)
        print(f"  centroid {sp}: n={mask.sum()}")

    # Sanity check: cosine distance often collapses for BirdNET (all positive features,
    # similar direction). Use Euclidean distance which handles magnitude differences.
    print(f"\n  sample cosine vs euclidean distance for first 3 recordings:")
    for i in range(3):
        sp = df.iloc[i]['common_name']
        c_d = cosine(X[i], centroids[sp])
        e_d = euclidean(X[i], centroids[sp])
        print(f"    rec[{i}] -> {sp} centroid: cosine={c_d:.6f}, euclidean={e_d:.4f}")

    # Use Euclidean distance to every species centroid
    print(f"\n  computing Euclidean distances to all {len(species_list)} species centroids ...")
    dist_cols = {f"d_to_{sp}": np.array([euclidean(X[i], centroids[sp]) for i in range(len(X))])
                 for sp in species_list}
    for k, v in dist_cols.items(): df[k] = v

    # Distance to own species centroid + nearest non-self
    own_dist = np.array([df[f"d_to_{df.iloc[i]['common_name']}"].iloc[i] for i in range(len(df))])
    df["d_to_self"] = own_dist

    # For each recording, find the NEAREST other species + distance
    nearest_other = []
    nearest_other_d = []
    for i in range(len(df)):
        sp = df.iloc[i]["common_name"]
        dists = {os: df.iloc[i][f"d_to_{os}"] for os in species_list if os != sp}
        nearest = min(dists, key=dists.get)
        nearest_other.append(nearest)
        nearest_other_d.append(dists[nearest])
    df["nearest_other_species"] = nearest_other
    df["d_to_nearest_other"] = nearest_other_d

    # Candidate mimicry: d_to_nearest_other < d_to_self
    df["mimicry_flag"] = df["d_to_nearest_other"] < df["d_to_self"]
    df["mimicry_margin"] = df["d_to_self"] - df["d_to_nearest_other"]  # positive = mimic-like

    n_mimicry = df["mimicry_flag"].sum()
    print(f"\n  candidate mimicry events: {n_mimicry} / {len(df)} ({100*n_mimicry/len(df):.1f}%)")
    print(f"\n  by species:")
    summary = df.groupby("common_name").agg(
        n=("id", "count"),
        n_mimicry=("mimicry_flag", "sum"),
        mean_d_to_self=("d_to_self", "mean"),
        max_mimicry_margin=("mimicry_margin", "max"),
    ).reset_index()
    summary["pct_mimicry"] = (100 * summary["n_mimicry"] / summary["n"]).round(1)
    print(summary.to_string(index=False))

    # Top 20 strongest mimicry candidates
    print(f"\n  === TOP 20 STRONGEST MIMICRY CANDIDATES (by margin) ===")
    top = df.sort_values("mimicry_margin", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"    XC{r['id']:<8s}  labeled={r['common_name']:<22s}  -> nearest={r['nearest_other_species']:<22s}  "
              f"d_self={r['d_to_self']:.4f}  d_other={r['d_to_nearest_other']:.4f}  margin={r['mimicry_margin']:.4f}")

    # Cross-species confusion matrix (what each species' mimics get labeled AS)
    print(f"\n  === CONFUSION TABLE: labeled X -> mimicry'd as Y ===")
    confusion = df[df["mimicry_flag"]].groupby(["common_name", "nearest_other_species"]).size().unstack(fill_value=0)
    print(confusion.to_string())

    # Save
    out_csv = OUT_DIR / "mimicry_candidates.csv"
    cols_out = ["id", "common_name", "ssp", "BCR_NAME", "lat_num", "lon_num",
                "d_to_self", "nearest_other_species", "d_to_nearest_other",
                "mimicry_margin", "mimicry_flag"] + [f"d_to_{s}" for s in species_list]
    df[cols_out].sort_values("mimicry_margin", ascending=False).to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv.relative_to(ROOT)}")

    # Markdown summary
    md = []
    md.append(f"# BirdNET Mimicry Detection — {time.strftime('%Y-%m-%d')}\n")
    md.append(f"Approach: cosine distance from each recording's BirdNET embedding to all 4 species' centroids. Flag where d(rec, own_species) > d(rec, other_species).\n")
    md.append(f"\n## Counts\n")
    md.append(f"- Total recordings: {len(df)}")
    md.append(f"- Mimicry candidates: {n_mimicry} ({100*n_mimicry/len(df):.1f}%)\n")
    md.append(f"\n## Per-species\n")
    md.append("| Species | n | n_mimicry | pct | mean d_to_self | max margin |")
    md.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        md.append(f"| {r['common_name']} | {r['n']} | {r['n_mimicry']} | {r['pct_mimicry']}% | {r['mean_d_to_self']:.4f} | {r['max_mimicry_margin']:.4f} |")
    md.append(f"\n## Top 20 strongest candidates\n")
    md.append("| XC ID | labeled | nearest other | d_self | d_other | margin |")
    md.append("|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        md.append(f"| XC{r['id']} | {r['common_name']} | {r['nearest_other_species']} | {r['d_to_self']:.4f} | {r['d_to_nearest_other']:.4f} | {r['mimicry_margin']:.4f} |")
    md.append(f"\n## Caveats\n")
    md.append(f"- BirdNET embedding distance is global, not phrase-level. A recording with even one mimicked phrase could pull the mean toward the mimic'd species.\n")
    md.append(f"- xeno-canto labels are submitter-asserted; some may be misidentified rather than mimicry.\n")
    md.append(f"- All 4 of our pilot species are non-mimic-known (chickadee, marsh wren, song sparrow, white-crowned sparrow). True mimics (mockingbird, catbird, lyrebird) aren't in this corpus. So any candidates here would be:\n  (a) misidentification, (b) hybrid/intergrade individuals, (c) genuine occasional cross-species learning.\n")
    (OUT_DIR / "mimicry_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  -> {OUT_DIR / 'mimicry_summary.md'}")


if __name__ == "__main__":
    main()
