"""DTW pairwise rhythm-trajectory distance test.

For each pair of 15 song-sparrow trajectories (5 per BCR × 3 BCRs):
  compute multivariate DTW distance on (onset_strength, ioi_rolling)
  (time_norm dropped — it's the parameter, not a feature)

Then split pairs into:
  - within-BCR (n = 5 choose 2 × 3 = 30 pairs)
  - between-BCR (n = remaining 75 pairs)

Test: is within-BCR distance distribution lower than between-BCR?
  - Mann-Whitney U (non-parametric, no normality assumption)
  - Report: median(within) vs median(between), U stat, p-value, effect size r
  - Per-BCR breakdown (does the pattern hold for each BCR or only some?)

Conclusion gates:
  IF p < 0.01 AND median(within) < median(between) → hypothesis SUPPORTED
                                                      (subject to multiple-comparison considerations)
  IF p > 0.05 → no detectable separation
  ELSE → suggestive, needs larger sample
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
IN_JSON = ROOT / "acoustic_features" / "rhythm_trajectories_v1.json"
OUT_JSON = ROOT / "viz" / "dtw_results.json"
OUT_MD = ROOT / "viz" / "dtw_results.md"


def dtw_multivariate(A, B):
    """Plain DTW on multivariate sequences A, B (shape: T_i, D).

    Returns DTW distance (path-length-normalized so different T's compare cleanly).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = len(A), len(B)
    INF = float("inf")
    # cost matrix
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = A[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    raw = D[n, m]
    # Path-length normalize: divide by (n+m) so trajectories of different lengths comparable
    return raw / (n + m)


def main():
    data = json.load(open(IN_JSON, encoding="utf-8"))
    trajs = data["trajectories"]
    print(f"=== DTW pairwise rhythm-trajectory test (n={len(trajs)}) ===")

    # Build (id, bcr, sequence) tuples; sequence = (onset_strength, ioi_rolling)
    seqs = []
    for t in trajs:
        x = np.array(t["trajectory"]["onset_strength"])
        y = np.array(t["trajectory"]["ioi_rolling"])
        seq = np.column_stack([x, y])  # shape (T, 2)
        seqs.append({"id": t["id"], "bcr": t["bcr"], "seq": seq, "n_frames": len(x)})

    # Compute pairwise DTW
    n = len(seqs)
    t0 = time.time()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_multivariate(seqs[i]["seq"], seqs[j]["seq"])
            pairs.append({
                "i": i, "j": j,
                "id_i": seqs[i]["id"], "id_j": seqs[j]["id"],
                "bcr_i": seqs[i]["bcr"], "bcr_j": seqs[j]["bcr"],
                "same_bcr": seqs[i]["bcr"] == seqs[j]["bcr"],
                "dtw": d,
            })
    elapsed = time.time() - t0
    print(f"  computed {len(pairs)} pairwise DTWs in {elapsed:.1f}s")

    df = pd.DataFrame(pairs)
    within = df[df["same_bcr"]]["dtw"].values
    between = df[~df["same_bcr"]]["dtw"].values
    print(f"\n  WITHIN-BCR pairs (n={len(within)}):")
    print(f"    median={np.median(within):.4f}  mean={np.mean(within):.4f}  std={np.std(within):.4f}")
    print(f"    min={np.min(within):.4f}  max={np.max(within):.4f}")
    print(f"\n  BETWEEN-BCR pairs (n={len(between)}):")
    print(f"    median={np.median(between):.4f}  mean={np.mean(between):.4f}  std={np.std(between):.4f}")
    print(f"    min={np.min(between):.4f}  max={np.max(between):.4f}")

    # Mann-Whitney U: is within < between?
    u_stat, p_value = mannwhitneyu(within, between, alternative="less")
    # Effect size: rank-biserial correlation r = 1 - 2U / (n1 * n2)
    r_rb = 1 - (2 * u_stat) / (len(within) * len(between))
    print(f"\n  === Mann-Whitney U (within < between, one-sided) ===")
    print(f"    U statistic: {u_stat:.0f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"    rank-biserial r: {r_rb:.3f}  (positive = within smaller than between)")

    # Per-BCR breakdown
    print(f"\n  === Per-BCR breakdown ===")
    per_bcr = {}
    for bcr in df["bcr_i"].unique():
        same = df[(df["bcr_i"] == bcr) & (df["bcr_j"] == bcr)]["dtw"].values
        diff = df[((df["bcr_i"] == bcr) ^ (df["bcr_j"] == bcr))]["dtw"].values
        # ^ is XOR; pairs where exactly one of i,j is bcr
        # Actually that's wrong for between-BCR involving this BCR. Let me re-derive.
        diff = df[((df["bcr_i"] == bcr) | (df["bcr_j"] == bcr)) & ~df["same_bcr"]]["dtw"].values
        if len(same) >= 3 and len(diff) >= 3:
            u, p = mannwhitneyu(same, diff, alternative="less")
            r = 1 - (2 * u) / (len(same) * len(diff))
            per_bcr[bcr] = {
                "n_within": len(same), "n_between": len(diff),
                "median_within": float(np.median(same)),
                "median_between": float(np.median(diff)),
                "p_value": float(p),
                "rank_biserial_r": float(r),
            }
            print(f"    {bcr}")
            print(f"      n_within={len(same)} n_between={len(diff)}")
            print(f"      median within={np.median(same):.4f}  between={np.median(diff):.4f}")
            print(f"      p={p:.4f}  r={r:.3f}")

    # Save
    df["dtw"] = df["dtw"].astype(float)
    results = {
        "test": "Mann-Whitney U one-sided (within < between) on DTW distances",
        "n_trajectories": len(seqs),
        "n_pairs_within": int(len(within)),
        "n_pairs_between": int(len(between)),
        "within_stats": {
            "median": float(np.median(within)), "mean": float(np.mean(within)),
            "std": float(np.std(within)), "min": float(np.min(within)),
            "max": float(np.max(within)),
        },
        "between_stats": {
            "median": float(np.median(between)), "mean": float(np.mean(between)),
            "std": float(np.std(between)), "min": float(np.min(between)),
            "max": float(np.max(between)),
        },
        "overall_mannwhitney_u": float(u_stat),
        "overall_p_value": float(p_value),
        "overall_rank_biserial_r": float(r_rb),
        "per_bcr": per_bcr,
        "pairs": df.to_dict(orient="records"),
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  -> {OUT_JSON}")

    # Markdown summary
    md = []
    md.append("# DTW Pairwise Rhythm-Trajectory Test\n")
    md.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("Test: within-BCR DTW distance < between-BCR DTW distance? Mann-Whitney U one-sided.\n")
    md.append(f"\n## Headline\n")
    md.append(f"- **Overall p = {p_value:.6f}**")
    md.append(f"- **Rank-biserial r = {r_rb:.3f}** (positive = within smaller, supports accent hypothesis)")
    md.append(f"- Within-BCR median: {np.median(within):.4f} (n={len(within)})")
    md.append(f"- Between-BCR median: {np.median(between):.4f} (n={len(between)})")
    md.append(f"- Difference: {(np.median(between) - np.median(within)):.4f}")
    md.append(f"\n## Verdict\n")
    if p_value < 0.01:
        md.append("**Hypothesis SUPPORTED at p < 0.01.** Within-BCR pairs are significantly more similar than between-BCR pairs in rhythm-anchored DTW space. First-pass evidence for regional rhythm accents in song sparrow.\n")
    elif p_value < 0.05:
        md.append("**Hypothesis suggestively supported (0.01 < p < 0.05).** Within-BCR similarity exceeds between-BCR but the signal is borderline. Larger sample or stricter feature axes recommended.\n")
    else:
        md.append("**Hypothesis not supported at this sample size (p >= 0.05).** Within-BCR vs between-BCR distance distributions don't separate cleanly. Possible: (a) need different rhythm features, (b) need finer geographic resolution than BCR, (c) sample too small.\n")
    md.append(f"\n## Per-BCR breakdown\n")
    md.append("| BCR | n_within | n_between | med_within | med_between | p | r |")
    md.append("|---|---|---|---|---|---|---|")
    for bcr, b in per_bcr.items():
        md.append(f"| {bcr} | {b['n_within']} | {b['n_between']} | {b['median_within']:.4f} | {b['median_between']:.4f} | {b['p_value']:.4f} | {b['rank_biserial_r']:.3f} |")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"  -> {OUT_MD}")


if __name__ == "__main__":
    main()
