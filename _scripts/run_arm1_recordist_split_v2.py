"""Phase 2 robustness axis 3 (substitute): recordist-source split within xeno-canto.

Macaulay source-swap is the canonical pre-reg axis 3, but requires Cornell
research access (gated, weeks-months). As a "best available substitute" we
stratify our existing xeno-canto recordings by recordist (xc `rec` field)
into prolific-recordist vs long-tail subsets. Refit Arm 1 on each, compare.

If the spectral signal holds when restricted to a DIFFERENT subset of
recordists (different mics, recording styles, locations they preferred),
it's evidence the signal isn't a single-recordist-equipment artifact.

This is logged as a pre-reg DEVIATION (substitute for canonical axis 3).
Macaulay swap remains queued for when institutional access lands.
"""
import os
RTOOLS = "C:/Users/Nate/.cmdstan/RTools40"
os.environ["PATH"] = f"{RTOOLS}/mingw64/bin;{RTOOLS}/usr/bin;" + os.environ.get("PATH", "")
import pathlib, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

ROOT = pathlib.Path(r"D:/Bird Song")
DESIGN = ROOT / "analysis" / "arm1_design_v2.parquet"
RAW    = ROOT / "range_data" / "xc_metadata_4species_conus.parquet"
STAN   = ROOT / "analysis" / "fit_arm1.stan"
OUT    = ROOT / "analysis" / "arm1_recordist_split_v2"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading design + recordist info ...")
df = pd.read_parquet(DESIGN)
raw = pd.read_parquet(RAW)
raw["id"] = raw["id"].astype(str)
df["id"] = df["id"].astype(str)
df = df.merge(raw[["id","rec"]], on="id", how="left")
print(f"  N: {len(df)}, recordists with >=1 rec: {df['rec'].nunique()}")

# Top recordist analysis
rec_counts = df["rec"].value_counts()
print(f"\n  top 20 recordists by recording count:")
print(rec_counts.head(20).to_string())

# Split: top-half-of-volume vs rest (by recordist contribution)
# Take top-N recordists who together hold ~50% of recordings
cumulative = rec_counts.cumsum()
half = len(df) / 2
n_top = int((cumulative <= half).sum()) + 1
top_recordists = rec_counts.head(n_top).index.tolist()
print(f"\n  top {n_top} recordists account for ~50% of recordings ({rec_counts.head(n_top).sum()} of {len(df)})")
df["recordist_group"] = np.where(df["rec"].isin(top_recordists), "top", "longtail")
print(f"\n  split sizes:")
print(df.groupby("recordist_group").size().to_string())

results = {}

for group in ["top", "longtail"]:
    print(f"\n=== Subset: {group} recordists ===")
    sub = df[df["recordist_group"] == group].copy()
    cell_n = sub.groupby("cell_id").size()
    sub = sub[sub["cell_id"].isin(cell_n[cell_n >= 2].index)].copy()
    if len(sub) < 50 or sub["cell_id"].nunique() < 5:
        print(f"  too few cells ({sub['cell_id'].nunique()}) or rows ({len(sub)}); skipping")
        continue
    sp_codes  = pd.Categorical(sub["common_name"]).codes + 1
    bcr_codes = pd.Categorical(sub["bcr_str"]).codes + 1
    cell_codes= pd.Categorical(sub["cell_id"]).codes + 1
    S = int(sp_codes.max()); B = int(bcr_codes.max()); C = int(cell_codes.max())
    print(f"  N={len(sub)}  S={S}  BCR={B}  cells={C}")

    model = CmdStanModel(stan_file=str(STAN))
    group_results = {}
    for outcome in ["d_spectral", "d_structural"]:
        y_raw = sub[outcome].values
        y_z = np.zeros_like(y_raw, dtype=float)
        for s_code in range(1, S+1):
            m = (sp_codes == s_code); v = y_raw[m]
            if np.std(v) > 1e-8: y_z[m] = (v - v.mean()) / v.std()
        t0 = time.time()
        fit = model.sample(
            data={
                "N": len(sub), "S": S, "B": B, "C": C,
                "species": sp_codes.tolist(),
                "bcr": bcr_codes.tolist(),
                "cell": cell_codes.tolist(),
                "y": y_z.tolist(),
            },
            chains=4, parallel_chains=4, iter_warmup=1000, iter_sampling=1000,
            adapt_delta=0.95, show_progress=False, seed=20260517,
        )
        post = fit.draws_pd()
        fc = post["frac_cell"]
        rec = {
            "group": group, "outcome": outcome,
            "n": int(len(sub)), "S": S, "B": B, "C": C,
            "frac_cell_mean": float(fc.mean()),
            "frac_cell_q025": float(fc.quantile(0.025)),
            "frac_cell_q975": float(fc.quantile(0.975)),
            "passes_5pct": bool(fc.quantile(0.025) > 0.05),
            "wall_sec": time.time() - t0,
        }
        print(f"  {outcome:<15}  frac_cell mean={rec['frac_cell_mean']:.3f} "
              f"95% CrI [{rec['frac_cell_q025']:.3f}, {rec['frac_cell_q975']:.3f}]  "
              f"passes 5%: {rec['passes_5pct']}")
        group_results[outcome] = rec
    results[group] = group_results

# Summary
print("\n=== RECORDIST-SPLIT SUMMARY ===")
for outcome in ["d_spectral", "d_structural"]:
    print(f"\n  {outcome}:")
    for group in ("top", "longtail"):
        if group in results and outcome in results[group]:
            r = results[group][outcome]
            print(f"    {group:<10} N={r['n']:<5} frac_cell={r['frac_cell_mean']:.3f} [{r['frac_cell_q025']:.3f}, {r['frac_cell_q975']:.3f}]  "
                  f"passes 5%: {r['passes_5pct']}")

with open(OUT / "recordist_split_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {OUT}/recordist_split_results.json")
