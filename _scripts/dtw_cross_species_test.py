"""Cross-species DTW test for the "affected by surrounding population" claim.

30 trajectories: 15 song sparrow + 15 marsh wren, both x 5 recordings x 3 BCRs.

For each pair (435 total), categorize:
  - WITHIN-SPECIES + SAME-BCR     (e.g., song-sparrow CA ↔ song-sparrow CA)
  - WITHIN-SPECIES + CROSS-BCR    (e.g., song-sparrow CA ↔ song-sparrow NE)
  - CROSS-SPECIES + SAME-BCR      (e.g., song-sparrow CA ↔ marsh-wren CA)
  - CROSS-SPECIES + CROSS-BCR     (e.g., song-sparrow CA ↔ marsh-wren NE)

The key test for the population-mediated-accent hypothesis:
  Is dist(CROSS-SPECIES + SAME-BCR) < dist(WITHIN-SPECIES + CROSS-BCR)?

If YES → birds in the same region share more rhythm than birds of the
         same species in different regions → REGIONAL ACOUSTIC
         ENVIRONMENT > SPECIES GENETICS for rhythm. Population matters.

If NO  → species-genetic dialect > regional convergence. Birds keep their
         species rhythm regardless of region.
"""
import pathlib, sys, io, json, time
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(r"D:/Bird Song")
SS_JSON = ROOT / "acoustic_features" / "rhythm_trajectories_v1.json"
MW_JSON = ROOT / "acoustic_features" / "rhythm_trajectories_marsh_wren_v1.json"
OUT_MD = ROOT / "viz" / "dtw_cross_species_results.md"


def dtw_multivariate(A, B):
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    n, m = len(A), len(B)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = A[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]) / (n + m)


def load_seqs(path, species):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for t in d["trajectories"]:
        x = np.array(t["trajectory"]["onset_strength"])
        y = np.array(t["trajectory"]["ioi_rolling"])
        out.append({"id": t["id"], "species": species, "bcr": t["bcr"],
                    "seq": np.column_stack([x, y]), "n_frames": len(x)})
    return out


def main():
    seqs = load_seqs(SS_JSON, "song_sparrow") + load_seqs(MW_JSON, "marsh_wren")
    print(f"=== Cross-species DTW test (n={len(seqs)}) ===")
    print(f"  song sparrow: {sum(1 for s in seqs if s['species']=='song_sparrow')}")
    print(f"  marsh wren:   {sum(1 for s in seqs if s['species']=='marsh_wren')}")

    t0 = time.time()
    pairs = []
    n = len(seqs)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_multivariate(seqs[i]["seq"], seqs[j]["seq"])
            same_species = seqs[i]["species"] == seqs[j]["species"]
            same_bcr = seqs[i]["bcr"] == seqs[j]["bcr"]
            cat = (
                "WS_SB" if (same_species and same_bcr) else
                "WS_DB" if (same_species and not same_bcr) else
                "DS_SB" if (not same_species and same_bcr) else
                "DS_DB"
            )
            pairs.append({"i": i, "j": j, "id_i": seqs[i]["id"], "id_j": seqs[j]["id"],
                          "sp_i": seqs[i]["species"], "sp_j": seqs[j]["species"],
                          "bcr_i": seqs[i]["bcr"], "bcr_j": seqs[j]["bcr"],
                          "category": cat, "dtw": d})
    print(f"  {len(pairs)} pairs computed in {time.time()-t0:.1f}s")

    df = pd.DataFrame(pairs)
    print(f"\n  === Distance distributions per category ===")
    cat_labels = {
        "WS_SB": "WITHIN-species + SAME-BCR (e.g., SS-CA ↔ SS-CA)",
        "WS_DB": "WITHIN-species + CROSS-BCR (e.g., SS-CA ↔ SS-NE)",
        "DS_SB": "CROSS-species + SAME-BCR (e.g., SS-CA ↔ MW-CA)",
        "DS_DB": "CROSS-species + CROSS-BCR (e.g., SS-CA ↔ MW-NE)",
    }
    summary = []
    for cat, label in cat_labels.items():
        sub = df[df["category"] == cat]["dtw"].values
        if len(sub) == 0: continue
        print(f"  {cat} ({label})")
        print(f"    n={len(sub)}; median={np.median(sub):.4f}; mean={np.mean(sub):.4f}; "
              f"std={np.std(sub):.4f}; range=[{sub.min():.4f}, {sub.max():.4f}]")
        summary.append({"cat": cat, "label": label, "n": len(sub),
                        "median": float(np.median(sub)), "mean": float(np.mean(sub)),
                        "std": float(np.std(sub))})

    # === Key test ===
    print(f"\n  === KEY TEST: cross-species + same-BCR  vs  within-species + cross-BCR ===")
    ds_sb = df[df["category"] == "DS_SB"]["dtw"].values
    ws_db = df[df["category"] == "WS_DB"]["dtw"].values
    print(f"  CROSS-SPECIES + SAME-BCR (DS_SB):  n={len(ds_sb)}  median={np.median(ds_sb):.4f}")
    print(f"  WITHIN-SPECIES + CROSS-BCR (WS_DB): n={len(ws_db)}  median={np.median(ws_db):.4f}")
    u, p = mannwhitneyu(ds_sb, ws_db, alternative="less")
    r = 1 - (2 * u) / (len(ds_sb) * len(ws_db))
    print(f"  Mann-Whitney U (DS_SB < WS_DB): U={u:.0f}  p={p:.6f}  rank-biserial r={r:.3f}")
    print()
    if p < 0.01:
        verdict = "POPULATION-MEDIATED ACCENTS SUPPORTED (p < 0.01). Different species in the same region are more similar than the same species in different regions."
    elif p < 0.05:
        verdict = "Suggestive (0.01 < p < 0.05). Cross-species same-BCR distances trend smaller than within-species cross-BCR, but the signal is borderline."
    elif r > 0:
        verdict = "Trend but not significant. DS_SB has lower median than WS_DB; need larger n."
    else:
        verdict = "REJECTED — species-genetic rhythm dominates over regional convergence. Same species across BCRs are still closer than different species in the same BCR."
    print(f"  VERDICT: {verdict}")

    # Secondary test
    print(f"\n  === SECONDARY TEST: within-species + same-BCR  vs  within-species + cross-BCR ===")
    ws_sb = df[df["category"] == "WS_SB"]["dtw"].values
    u2, p2 = mannwhitneyu(ws_sb, ws_db, alternative="less")
    r2 = 1 - (2 * u2) / (len(ws_sb) * len(ws_db))
    print(f"  WS_SB: n={len(ws_sb)} median={np.median(ws_sb):.4f}")
    print(f"  WS_DB: n={len(ws_db)} median={np.median(ws_db):.4f}")
    print(f"  Mann-Whitney U (WS_SB < WS_DB): U={u2:.0f}  p={p2:.6f}  r={r2:.3f}")

    # Markdown
    md = []
    md.append("# Cross-Species DTW Test — Population-Mediated Accent\n")
    md.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"\n## Sample\n")
    md.append(f"- 15 song sparrows + 15 marsh wrens, both spanning 3 BCRs (5 per BCR per species)")
    md.append(f"- {len(pairs)} pairwise DTW distances on rhythm-anchored trajectories")
    md.append(f"\n## Category medians\n")
    md.append("| Category | label | n | median | mean | std |")
    md.append("|---|---|---|---|---|---|")
    for s in summary:
        md.append(f"| {s['cat']} | {s['label']} | {s['n']} | {s['median']:.4f} | {s['mean']:.4f} | {s['std']:.4f} |")
    md.append(f"\n## Key test\n")
    md.append(f"Hypothesis: if accents are population-mediated, then CROSS-species + SAME-BCR < WITHIN-species + CROSS-BCR.")
    md.append(f"- Cross-species same-BCR median: {np.median(ds_sb):.4f} (n={len(ds_sb)})")
    md.append(f"- Within-species cross-BCR median: {np.median(ws_db):.4f} (n={len(ws_db)})")
    md.append(f"- Mann-Whitney p={p:.6f}, rank-biserial r={r:.3f}")
    md.append(f"\n**VERDICT:** {verdict}\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  -> {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
