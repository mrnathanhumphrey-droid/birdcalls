"""Phase 2 robustness axis 1: 5-fold cell-stratified sample resampling.

Per pre-reg: SEED = 20260616, k = 5. Each fold holds out a stratified
sample of recordings (stratified by species × BCR × dialect_cluster cell).
Re-fits the v0.2 Arm 1 model on the 4 other folds, records frac_cell mean
+ 95% CrI per outcome per fold.

Pre-reg verdict rule for the sample-resampling axis:
  STRONG: spectral outcome PASS (frac_cell q025 > 0.05) in >= 4 of 5 folds.
  PARTIAL: PASS in 3 of 5.
  WEAK: PASS in <= 2.
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
DESIGN = ROOT / "analysis" / "arm1_design.parquet"
STAN   = ROOT / "analysis" / "fit_arm1.stan"
OUT    = ROOT / "analysis" / "arm1_5fold"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading design + stratifying ...")
df = pd.read_parquet(DESIGN)
cell_n = df.groupby("cell_id").size()
df = df[df["cell_id"].isin(cell_n[cell_n >= 2].index)].copy()
N = len(df)
print(f"  N = {N}, cells = {df['cell_id'].nunique()}")

SEED = 20260616
K = 5
rng = np.random.default_rng(SEED)
df["fold"] = -1
for cid, sub in df.groupby("cell_id"):
    perm = rng.permutation(len(sub))
    folds = np.array([(i % K) + 1 for i in range(len(sub))])
    folds = folds[perm]
    df.loc[sub.index, "fold"] = folds
print("\n  per-fold sizes:")
print(df.groupby("fold").size().to_string())

print("\n[2] Compiling Stan model ...")
model = CmdStanModel(stan_file=str(STAN))

# Save fold assignments for audit (per pre-reg immutability)
df[["id","cell_id","fold"]].to_csv(OUT / "fold_assignments.csv", index=False)
print(f"  fold_assignments saved (immutable)")

all_results = []
for outcome in ["d_spectral", "d_structural"]:
    print(f"\n=== Outcome: {outcome} ===")
    for k_held in range(1, K+1):
        train = df[df["fold"] != k_held].copy()
        # Re-encode codes after dropping
        sp_codes  = pd.Categorical(train["common_name"]).codes + 1
        bcr_codes = pd.Categorical(train["bcr_str"]).codes + 1
        cell_codes= pd.Categorical(train["cell_id"]).codes + 1
        S = int(sp_codes.max()); B = int(bcr_codes.max()); C = int(cell_codes.max())
        y_raw = train[outcome].values
        y_z = np.zeros_like(y_raw, dtype=float)
        for s_code in range(1, S+1):
            m = (sp_codes == s_code); v = y_raw[m]
            if np.std(v) > 1e-8: y_z[m] = (v - v.mean()) / v.std()
        t0 = time.time()
        fit = model.sample(
            data={
                "N": len(train), "S": S, "B": B, "C": C,
                "species": sp_codes.tolist(),
                "bcr": bcr_codes.tolist(),
                "cell": cell_codes.tolist(),
                "y": y_z.tolist(),
            },
            chains=4, parallel_chains=4,
            iter_warmup=1000, iter_sampling=1000,
            adapt_delta=0.95, max_treedepth=10,
            show_progress=False, seed=SEED + k_held,
        )
        post = fit.draws_pd()
        fc = post["frac_cell"]
        rec = {
            "outcome": outcome, "fold_held": k_held, "n_train": int(len(train)),
            "frac_cell_mean": float(fc.mean()),
            "frac_cell_q025": float(fc.quantile(0.025)),
            "frac_cell_q975": float(fc.quantile(0.975)),
            "passes_5pct":  bool(fc.quantile(0.025) > 0.05),
            "ci_clean_pos": bool(fc.quantile(0.025) > 0),
            "wall_sec": time.time() - t0,
        }
        all_results.append(rec)
        print(f"  fold {k_held} held: frac_cell mean={rec['frac_cell_mean']:.3f} "
              f"95% CrI [{rec['frac_cell_q025']:.3f}, {rec['frac_cell_q975']:.3f}]  "
              f"passes 5%: {rec['passes_5pct']}")

# Summary
print("\n=== SAMPLE-RESAMPLING SUMMARY ===")
for outcome in ["d_spectral", "d_structural"]:
    subset = [r for r in all_results if r["outcome"] == outcome]
    passes = sum(r["passes_5pct"] for r in subset)
    print(f"  {outcome:<15} passes 5% in {passes}/5 folds")
    means = [r["frac_cell_mean"] for r in subset]
    print(f"    fold means: {[f'{m:.3f}' for m in means]}")

with open(OUT / "5fold_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nsaved {OUT}/5fold_results.json")
