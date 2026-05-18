"""Random-cluster-labels sanity test.

If frac_cell ≈ 0.19 holds even with RANDOM cluster assignments, then
cell-partial-pooling is fitting degrees-of-freedom rather than real
dialect structure. If frac_cell drops to ~0 with random labels, the v2
cell partition captures genuine signal.

Same Stan model as Arm 1 main fit. Just shuffle dialect_cluster within each
species before building cells. Run 5 random seeds to estimate distribution.
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
STAN   = ROOT / "analysis" / "fit_arm1.stan"
OUT    = ROOT / "analysis" / "arm1_random_labels_test"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading v2 design ...")
df_base = pd.read_parquet(DESIGN)
print(f"  rows: {len(df_base)}  v2 cells: {df_base['cell_id'].nunique()}")

print("\n[2] Compiling Stan model ...")
model = CmdStanModel(stan_file=str(STAN))

n_seeds = 5
results = []
rng = np.random.default_rng(20260518)

for seed_i in range(n_seeds):
    df = df_base.copy()
    # Shuffle dialect_cluster within each species (preserves per-species cluster count)
    for sp, sub in df.groupby("common_name"):
        labels = df.loc[sub.index, "dialect_cluster"].values.copy()
        rng.shuffle(labels)
        df.loc[sub.index, "dialect_cluster"] = labels
    df["dc_str"] = df["dialect_cluster"].astype(str)
    df["cell_id"] = df["common_name"] + "::" + df["bcr_str"] + "::" + df["dc_str"]
    cell_n = df.groupby("cell_id").size()
    keep_cells = cell_n[cell_n >= 2].index
    df = df[df["cell_id"].isin(keep_cells)].copy()

    species_codes = pd.Categorical(df["common_name"]).codes + 1
    bcr_codes     = pd.Categorical(df["bcr_str"]).codes + 1
    cell_codes    = pd.Categorical(df["cell_id"]).codes + 1
    S, B, C, N = int(species_codes.max()), int(bcr_codes.max()), int(cell_codes.max()), len(df)
    print(f"\n[3] Seed {seed_i+1}/{n_seeds}: N={N}, cells={C}")

    y_raw = df["d_spectral"].values
    y_z = np.zeros_like(y_raw, dtype=float)
    for s_code in range(1, S+1):
        m = (species_codes == s_code); v = y_raw[m]
        if np.std(v) > 1e-8: y_z[m] = (v - v.mean()) / v.std()

    t0 = time.time()
    fit = model.sample(
        data={"N": N, "S": S, "B": B, "C": C,
              "species": species_codes.tolist(), "bcr": bcr_codes.tolist(),
              "cell": cell_codes.tolist(), "y": y_z.tolist()},
        chains=4, parallel_chains=4,
        iter_warmup=600, iter_sampling=600,
        adapt_delta=0.95, max_treedepth=10,
        show_progress=False, seed=20260518 + seed_i,
    )
    post = fit.draws_pd()
    fc = post["frac_cell"]
    res = {"seed": seed_i, "N": N, "C": C,
           "frac_cell_mean": float(fc.mean()),
           "frac_cell_q025": float(fc.quantile(0.025)),
           "frac_cell_q975": float(fc.quantile(0.975)),
           "passes_5pct": bool(fc.quantile(0.025) > 0.05),
           "wall_sec": time.time() - t0}
    print(f"  d_spectral random seed {seed_i}: frac_cell mean={res['frac_cell_mean']:.3f} "
          f"CrI=[{res['frac_cell_q025']:.3f}, {res['frac_cell_q975']:.3f}]  "
          f"passes 5%: {res['passes_5pct']}")
    results.append(res)

# Summary
means = [r["frac_cell_mean"] for r in results]
passes = [r["passes_5pct"] for r in results]
print(f"\n=== Random-labels d_spectral summary ({n_seeds} seeds) ===")
print(f"  frac_cell mean: {np.mean(means):.3f}  (range {min(means):.3f}..{max(means):.3f})")
print(f"  passes 5% in: {sum(passes)}/{n_seeds} seeds")
print(f"  TRUE v2 dialect_cluster frac_cell: 0.198 (PASS)")
if np.mean(means) > 0.10:
    print(f"\n  CONCERN: random labels also yield meaningful frac_cell.")
    print(f"  Cell-partial-pooling spectral signal may be degrees-of-freedom artifact.")
elif np.mean(means) < 0.05:
    print(f"\n  RELIEF: random labels yield near-zero frac_cell.")
    print(f"  The v2 cell partition captures real spectral structure.")
else:
    print(f"\n  MIXED: random labels are between null and v2 truth. Some structure exists in v2 partition that random can't recover.")

(OUT / "random_labels_results.json").write_text(
    json.dumps({"results": results, "v2_truth_frac_cell": 0.198,
                "random_mean_frac_cell": float(np.mean(means)),
                "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
