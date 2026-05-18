"""DTW on phrase-template sequences — Angle 2 test.

Per-recording phrase template: ordered sequence of phrase feature vectors
[(duration, n_syllables, mean_iio, mean_intensity, max_intensity), ...]

Pairwise multivariate DTW on these sequences, then the same
4-category test (within/cross species × within/cross BCR) as the per-frame
DTW pass. Phrase-level is more aggregated → less frame-noise, more
sensitive to stereotyped structure.
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
IN_JSON = ROOT / "acoustic_features" / "phrase_templates_v1.json"
OUT_MD = ROOT / "viz" / "dtw_phrase_template_results.md"

FEATURE_KEYS = ["duration", "n_syllables", "mean_iio", "mean_intensity", "max_intensity"]


def standardize(seqs):
    """Z-score each feature dim across all recordings."""
    all_vals = np.vstack([s["arr"] for s in seqs])
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0) + 1e-9
    for s in seqs:
        s["arr"] = (s["arr"] - mean) / std
    return seqs


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


def main():
    data = json.load(open(IN_JSON, encoding="utf-8"))
    templates = data["templates"]
    print(f"=== Phrase-template DTW test (n={len(templates)}) ===")

    seqs = []
    for t in templates:
        if t["n_phrases"] < 2: continue  # need ≥2 phrases for DTW
        arr = np.array([[p[k] for k in FEATURE_KEYS] for p in t["phrases"]], dtype=float)
        seqs.append({"id": t["id"], "species": t["species"].replace(" ", "_"),
                     "bcr": t["bcr"], "arr": arr, "n_phrases": t["n_phrases"]})
    print(f"  usable (n_phrases >= 2): {len(seqs)}")

    seqs = standardize(seqs)

    t0 = time.time()
    pairs = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            d = dtw_multivariate(seqs[i]["arr"], seqs[j]["arr"])
            ssp = seqs[i]["species"] == seqs[j]["species"]
            sbcr = seqs[i]["bcr"] == seqs[j]["bcr"]
            cat = ("WS_SB" if (ssp and sbcr) else
                   "WS_DB" if (ssp and not sbcr) else
                   "DS_SB" if (not ssp and sbcr) else
                   "DS_DB")
            pairs.append({"id_i": seqs[i]["id"], "id_j": seqs[j]["id"],
                          "sp_i": seqs[i]["species"], "sp_j": seqs[j]["species"],
                          "bcr_i": seqs[i]["bcr"], "bcr_j": seqs[j]["bcr"],
                          "category": cat, "dtw": d})
    print(f"  {len(pairs)} pairs computed in {time.time()-t0:.1f}s")

    df = pd.DataFrame(pairs)
    cat_labels = {
        "WS_SB": "WITHIN-species + SAME-BCR",
        "WS_DB": "WITHIN-species + CROSS-BCR",
        "DS_SB": "CROSS-species + SAME-BCR",
        "DS_DB": "CROSS-species + CROSS-BCR",
    }
    print(f"\n  === Distance distributions per category ===")
    cat_data = {}
    for cat, label in cat_labels.items():
        sub = df[df["category"] == cat]["dtw"].values
        cat_data[cat] = sub
        if len(sub) == 0: continue
        print(f"  {cat} ({label}): n={len(sub)}; median={np.median(sub):.4f}; "
              f"mean={np.mean(sub):.4f}; std={np.std(sub):.4f}")

    # Key test: DS_SB < WS_DB?
    u, p = mannwhitneyu(cat_data["DS_SB"], cat_data["WS_DB"], alternative="less")
    r = 1 - (2 * u) / (len(cat_data["DS_SB"]) * len(cat_data["WS_DB"]))
    print(f"\n  KEY TEST (DS_SB < WS_DB): p={p:.6f}  r={r:.3f}")

    # Secondary: WS_SB < WS_DB?
    u2, p2 = mannwhitneyu(cat_data["WS_SB"], cat_data["WS_DB"], alternative="less")
    r2 = 1 - (2 * u2) / (len(cat_data["WS_SB"]) * len(cat_data["WS_DB"]))
    print(f"  SECONDARY (WS_SB < WS_DB): p={p2:.6f}  r={r2:.3f}")

    # Also: WS < DS regardless of BCR (species-genetic signal)
    ws = np.concatenate([cat_data["WS_SB"], cat_data["WS_DB"]])
    ds = np.concatenate([cat_data["DS_SB"], cat_data["DS_DB"]])
    u3, p3 = mannwhitneyu(ws, ds, alternative="less")
    r3 = 1 - (2 * u3) / (len(ws) * len(ds))
    print(f"  TERTIARY (within-species < cross-species, any BCR): p={p3:.6f}  r={r3:.3f}")

    # Markdown
    md = []
    md.append("# Phrase-Template DTW Test (Angle 2)\n")
    md.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n## Sample\n")
    md.append(f"- {len(seqs)} recordings with ≥2 phrases each (out of {len(templates)})")
    md.append(f"- Phrase features standardized (z-score) across all recordings, 5 dims")
    md.append(f"- Phrase gap threshold: 0.5s\n")
    md.append(f"\n## Category medians\n")
    md.append("| Category | label | n | median | mean | std |")
    md.append("|---|---|---|---|---|---|")
    for cat, label in cat_labels.items():
        s = cat_data[cat]
        if len(s) == 0: continue
        md.append(f"| {cat} | {label} | {len(s)} | {np.median(s):.4f} | {np.mean(s):.4f} | {np.std(s):.4f} |")
    md.append(f"\n## Tests\n")
    md.append(f"- KEY (population-mediated accent): cross-species same-BCR < within-species cross-BCR? **p={p:.4f}**, r={r:.3f}")
    md.append(f"- Within-species regional accent: WS-same-BCR < WS-cross-BCR? **p={p2:.4f}**, r={r2:.3f}")
    md.append(f"- Species genetic signal: within-species < cross-species (any BCR)? **p={p3:.4f}**, r={r3:.3f}")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  -> {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
