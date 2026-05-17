"""Sainburg-2020-style industry-baseline head-to-head.

Sainburg et al. 2020 *PLOS Comp Bio*'s pipeline: UMAP+HDBSCAN clusters on
spectral features. We use the same dialect_cluster labels (discovered via
UMAP-8d + HDBSCAN on BirdNET embeddings — structurally equivalent to Sainburg)
as a SOLO predictor with no species/BCR layers.

The test: does our full hierarchical model (species + BCR + cell) explain
MORE variance than the unsupervised cluster labels alone?

Reports per outcome:
  frac_dc_sainburg: variance explained by dialect_cluster alone
  frac_cell_ours: variance explained by cell residual layer in our full model
  delta = frac_cell_ours - frac_dc_sainburg
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
STAN_S = ROOT / "analysis" / "fit_arm1_sainburg.stan"
STAN_F = ROOT / "analysis" / "fit_arm1.stan"
OUT    = ROOT / "analysis" / "sainburg_baseline"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading Arm 1 design ...")
df = pd.read_parquet(DESIGN)
cell_n = df.groupby("cell_id").size()
df = df[df["cell_id"].isin(cell_n[cell_n >= 2].index)].copy()
print(f"  N = {len(df)}, cells = {df['cell_id'].nunique()}, dialects = {df['dialect_cluster'].nunique()}")

# Sainburg's unit is dialect_cluster within species, so encode species×dialect
df["sain_label"] = df["common_name"] + "_dc" + df["dialect_cluster"].astype(str)
dc_codes = pd.Categorical(df["sain_label"]).codes + 1
sp_codes  = pd.Categorical(df["common_name"]).codes + 1
bcr_codes = pd.Categorical(df["bcr_str"]).codes + 1
cell_codes= pd.Categorical(df["cell_id"]).codes + 1
S = int(sp_codes.max()); B = int(bcr_codes.max()); C = int(cell_codes.max())
D = int(dc_codes.max())
print(f"  S={S}, BCR={B}, cells={C}, sainburg_labels={D}")

model_s = CmdStanModel(stan_file=str(STAN_S))
model_f = CmdStanModel(stan_file=str(STAN_F))

results = {}
for outcome in ["d_spectral", "d_structural"]:
    print(f"\n=== Outcome: {outcome} ===")
    y_raw = df[outcome].values
    y_z = np.zeros_like(y_raw, dtype=float)
    for s_code in range(1, S+1):
        m = (sp_codes == s_code); v = y_raw[m]
        if np.std(v) > 1e-8: y_z[m] = (v - v.mean()) / v.std()
    # Sainburg baseline
    print("  Fitting Sainburg-only baseline ...")
    fit_s = model_s.sample(
        data={"N": len(df), "D": D, "dc": dc_codes.tolist(), "y": y_z.tolist()},
        chains=4, parallel_chains=4, iter_warmup=1000, iter_sampling=1000,
        adapt_delta=0.95, show_progress=False, seed=20260517,
    )
    post_s = fit_s.draws_pd()
    fc_s = post_s["frac_dc"]
    # Full model (our method)
    print("  Fitting our full hierarchical (species + BCR + cell) ...")
    fit_f = model_f.sample(
        data={
            "N": len(df), "S": S, "B": B, "C": C,
            "species": sp_codes.tolist(),
            "bcr": bcr_codes.tolist(),
            "cell": cell_codes.tolist(),
            "y": y_z.tolist(),
        },
        chains=4, parallel_chains=4, iter_warmup=1000, iter_sampling=1000,
        adapt_delta=0.95, show_progress=False, seed=20260517,
    )
    post_f = fit_f.draws_pd()
    fc_f = post_f["frac_cell"]
    fb_f = post_f["frac_additive_baseline"]
    # Also extract: full model's TOTAL non-residual variance share (sp + bcr + cell)
    full_explained = post_f["frac_additive_baseline"] + post_f["frac_cell"]

    rec = {
        "outcome": outcome,
        "sainburg_frac_dc_mean": float(fc_s.mean()),
        "sainburg_frac_dc_q025": float(fc_s.quantile(0.025)),
        "sainburg_frac_dc_q975": float(fc_s.quantile(0.975)),
        "ours_frac_cell_mean": float(fc_f.mean()),
        "ours_frac_cell_q025": float(fc_f.quantile(0.025)),
        "ours_frac_cell_q975": float(fc_f.quantile(0.975)),
        "ours_frac_baseline_mean": float(fb_f.mean()),
        "ours_total_explained_mean": float(full_explained.mean()),
        "ours_total_explained_q025": float(full_explained.quantile(0.025)),
        "ours_total_explained_q975": float(full_explained.quantile(0.975)),
        "delta_ours_minus_sainburg": float(full_explained.mean() - fc_s.mean()),
    }
    print(f"  Sainburg (dialect_cluster only): frac_dc mean={rec['sainburg_frac_dc_mean']:.3f} "
          f"95% CrI [{rec['sainburg_frac_dc_q025']:.3f}, {rec['sainburg_frac_dc_q975']:.3f}]")
    print(f"  Ours (sp+BCR+cell): cell-only mean={rec['ours_frac_cell_mean']:.3f}, "
          f"total explained (sp+BCR+cell) mean={rec['ours_total_explained_mean']:.3f}")
    print(f"  Delta (ours TOTAL minus Sainburg): {rec['delta_ours_minus_sainburg']:+.3f}")
    results[outcome] = rec

with open(OUT / "sainburg_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}/sainburg_results.json")
