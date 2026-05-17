"""Arm 2 Stan fits — same model structure as Arm 1 but baseline is
(species + community) instead of (species + BCR). Cell = (community ×
spectral_band × time_of_day).

Decision rule: cell partial-pooling adds >= 5% additional variance on
BOTH outcomes (spectral niche, structural niche) with 95% CrI clean +.
"""
import os
RTOOLS = "C:/Users/Nate/.cmdstan/RTools40"
os.environ["PATH"] = f"{RTOOLS}/mingw64/bin;{RTOOLS}/usr/bin;" + os.environ.get("PATH", "")

import pathlib, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

ROOT   = pathlib.Path(r"D:/Bird Song")
DESIGN = ROOT / "analysis" / "arm2_design.parquet"
STAN   = ROOT / "analysis" / "fit_arm1.stan"  # reuse — same structure
OUT    = ROOT / "analysis" / "arm2_fits"
OUT.mkdir(parents=True, exist_ok=True)

print("[1] Loading Arm 2 design ...")
df = pd.read_parquet(DESIGN)
print(f"  rows: {len(df)}  cells: {df['cell_id'].nunique()}  communities: {df['h3'].nunique()}")

species_codes = pd.Categorical(df["common_name"]).codes + 1
comm_codes    = pd.Categorical(df["h3"]).codes + 1
cell_codes    = pd.Categorical(df["cell_id"]).codes + 1
S = int(species_codes.max())
B = int(comm_codes.max())   # reuse B var for community
C = int(cell_codes.max())
N = len(df)
print(f"  S={S}, communities={B}, C={C}, N={N}")

print("\n[2] Compiling Stan model ...")
model = CmdStanModel(stan_file=str(STAN))
print("  compiled.")

results = {}
for outcome in ["d_spectral_niche", "d_structural_niche"]:
    print(f"\n[3] Fitting outcome: {outcome}")
    y_raw = df[outcome].values
    y_z = np.zeros_like(y_raw, dtype=float)
    for s_code in range(1, S+1):
        m = (species_codes == s_code)
        v = y_raw[m]
        if np.std(v) > 1e-8:
            y_z[m] = (v - v.mean()) / v.std()
    t0 = time.time()
    fit = model.sample(
        data={
            "N": N, "S": S, "B": B, "C": C,
            "species": species_codes.tolist(),
            "bcr": comm_codes.tolist(),  # baseline grouping = community
            "cell": cell_codes.tolist(),
            "y": y_z.tolist(),
        },
        chains=4, parallel_chains=4,
        iter_warmup=1000, iter_sampling=1000,
        adapt_delta=0.95, max_treedepth=10,
        show_progress=False, seed=20260516,
    )
    print(f"  sampled in {time.time()-t0:.1f}s")
    diag = fit.diagnose()
    print(f"  diagnostics:\n    " + str(diag).replace("\n","\n    ")[:500])
    post = fit.draws_pd()
    frac_cell = post["frac_cell"]
    frac_base = post["frac_additive_baseline"]
    verdict = {
        "outcome": outcome,
        "frac_cell_mean": float(frac_cell.mean()),
        "frac_cell_q025": float(frac_cell.quantile(0.025)),
        "frac_cell_q975": float(frac_cell.quantile(0.975)),
        "ci_clean_positive": bool(frac_cell.quantile(0.025) > 0),
        "passes_5pct": bool(frac_cell.quantile(0.025) > 0.05),
        "frac_additive_mean": float(frac_base.mean()),
        "sigma_y_mean": float(post["sigma_y"].mean()),
        "sigma_cell_mean": float(post["sigma_cell"].mean()),
        "N": N, "S": S, "communities": B, "C": C,
    }
    print(f"  frac_cell mean={verdict['frac_cell_mean']:.3f} "
          f"95% CrI [{verdict['frac_cell_q025']:.3f}, {verdict['frac_cell_q975']:.3f}]")
    print(f"  passes 5% + CI rule: {verdict['passes_5pct']}")
    results[outcome] = verdict
    with open(OUT / f"verdict_{outcome}.json", "w") as f:
        json.dump(verdict, f, indent=2)

both_pass = all(r["passes_5pct"] for r in results.values())
any_pass  = any(r["passes_5pct"] for r in results.values())
disposition = "STRONG_REPLICATION" if both_pass else ("PARTIAL" if any_pass else "FAIL")
final = {"disposition": disposition, "by_outcome": results}
with open(OUT / "arm2_disposition.json", "w") as f:
    json.dump(final, f, indent=2)
print(f"\n=== ARM 2 DISPOSITION: {disposition} ===")
