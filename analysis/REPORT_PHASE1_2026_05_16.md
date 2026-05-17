# Bird Song Substrate — Phase 1 Report

**Date:** 2026-05-16
**Pre-registration:** locked at commit `aa38aea` on github.com/mrnathanhumphrey-droid/birdcalls

## Headline disposition

| Arm | Outcome | frac_cell mean | 95% CrI | Pre-reg PASS? |
|---|---|---|---|---|
| **Arm 1** (dialect-geography) | Spectral (BirdNET-embed dist) | **0.191** | [0.131, 0.251] | **✓ PASS** |
| **Arm 1** (dialect-geography) | Structural (z-Euclidean) | 0.032 | [0.004, 0.069] | ✗ FAIL (CI clean+ but mean <5%) |
| **Arm 2** (acoustic-niche-partition) | Spectral niche (centroid z-dist) | **0.209** | [0.068, 0.409] | **✓ PASS** |
| **Arm 2** (acoustic-niche-partition) | Structural niche (z-Euclidean) | 0.039 | [0.000, 0.113] | ✗ FAIL (CI grazes 0) |

**Arm 1 disposition: PARTIAL_REPLICATION** — spectral PASS, structural FAIL.
**Arm 2 disposition: PARTIAL_REPLICATION** — spectral PASS, structural FAIL.
**Joint Phase 1 disposition: PARTIAL on both arms; spectral signal load-bearing.**

## Scientific reading

Both arms test the same methodological claim — that the partial-pooling-of-residual-classes framework (Paper 6 corpus method) recovers acoustic residual structure beyond what additive baselines (species + region, or species + community) explain.

**Both arms agree on the same pattern:**
- ~19–21% of acoustic variance is captured by the cell residual layer in **spectral** features
- ~3–4% is captured in **structural/syntactic** features

This convergence across two independent operationalizations is the substantive finding: **bioacoustic differentiation, whether by dialect geography or sympatric acoustic-niche partition, operates primarily on spectral position — which syllables/sounds are used — rather than on syntactic structure — how those sounds are combined into sequences.**

This is consistent with Marler's classical white-crowned sparrow dialect literature: adjacent dialect populations differ in which syllable variants they sing, not in the rules of song construction. The same logic appears to apply to acoustic-niche partition: sympatric species partition the spectral niche, but the *syntax* of their songs (entropy, LZ complexity, inventory size) doesn't carry the partition signal at the cell-pooling resolution we tested.

## What survives, what doesn't

**Survives (PASS criterion: ≥5% additional variance + 95% CrI clean positive on the outcome):**
- Cell partial-pooling on BirdNET-embedding distance, dialect-geography cells (Arm 1)
- Cell partial-pooling on spectral-centroid niche distance, acoustic-niche cells (Arm 2)

**Doesn't survive at locked threshold:**
- Cell partial-pooling on structural features (syllable inventory size, transition Shannon entropy, Lempel-Ziv complexity, normalized LZ ratio) on either arm

**Honest qualifications:**
- Structural Arm 2 had 4.8% divergent transitions (borderline). Needs reparameterization (non-centered prior on cell-level sigma, or tighter half-normal prior on σ_y) before being final.
- Structural feature set (5 dims) may still be too coarse. Pre-reg locked these 5 specifically; deviation would require pre-reg amendment + log.

## Methods

### Data pipeline (deterministic, all pre-reg compliant)

1. **xeno-canto v3 API**: pull 2,171 records for 4 species (white-crowned sparrow, song sparrow, Carolina chickadee, marsh wren) in CONUS
2. **Filter** to song-type recordings with valid coords + quality ∈ {A,B,C} → 1,452
3. **BCR spatial join** (USFWS Bird Conservation Regions, USDA Forest Service RDS-2021-0001 layer) → 1,344 cell-eligible recordings across 34 NA BCRs
4. **Fetch audio**: full 1,344 mp3/wav files (~3.1 GB, ~1 hr at 1.5s throttle)
5. **SoundPlot feature extraction** (Mehdi et al. 2026 arXiv 2601.12752): 189 spectral + temporal features per recording, 60-sec cap, 12 workers, 3.4 hr wall
6. **BirdNET v2.4 embeddings**: 1024-dim mean + 1024-dim std per recording, 60-sec cap, 5.2 min wall
7. **Syllable-level structural pipeline**: librosa onset-segment → per-syllable MFCC-13 → UMAP-to-8d → HDBSCAN (min_cluster_size=15) per species → 191/828/1076/681 syllable types per species → per-recording inventory + transition entropy + LZ complexity
8. **Within-species dialect-cluster discovery**: UMAP-to-8d → HDBSCAN on BirdNET embedding-means → 14 dialect clusters across 4 species (CACH:2, MAWR:2, SOSP:6, WCSP:4)

### Arm 1 model

Hierarchical Gaussian regression in Stan:
```
y[n] = mu + alpha_sp[species[n]] + alpha_bcr[BCR[n]] + alpha_cell[cell[n]] + eps[n]
alpha_sp ~ N(0, sigma_sp);  alpha_bcr ~ N(0, sigma_bcr);  alpha_cell ~ N(0, sigma_cell)
```
Cells = (species × BCR × dialect_cluster). Test statistic: `frac_cell = var(alpha_cell) / (var_sp + var_bcr + var_cell + sigma_y^2)`.

Outcomes z-scored within species (so different scales don't dominate). 4 chains × 1000 warmup + 1000 sampling, adapt_delta 0.95, seed 20260516.

### Arm 2 model

Identical Stan structure, baseline rewired to `species + community` (community = H3 hex level 5 from recording lat/lon, filtered to hexes with ≥2 pilot species). Cell = (community × spectral_band × time_of_day).

- spectral_band: low (<2kHz), mid (2-6kHz), high (>6kHz) from SoundPlot `spectral_centroid_mean`
- time_of_day: dawn (4-8am), day (8am-6pm), dusk_night (else) from xc `time` field

After filters: 499 recordings across 113 cells (n≥2 threshold).

### Outcomes

- **d_spectral** (Arm 1): Euclidean distance from species centroid in 1024-dim BirdNET embedding-mean space
- **d_structural** (Arm 1): z-Euclidean across 5 structural features (n_syllables, syllable_inventory_size, transition_entropy, lz_complexity, lz_ratio)
- **d_spectral_niche** (Arm 2): |z-score of spectral_centroid_mean within species|
- **d_structural_niche** (Arm 2): z-Euclidean across 4 structural features within species

## Files

```
analysis/
├── arm1_design.parquet             (1,344 recordings × design fields)
├── arm2_design.parquet             (499 recordings × design fields)
├── fit_arm1.stan                   (hierarchical Gaussian + variance decomp)
├── arm1_fits/
│   ├── verdict_d_spectral.json
│   ├── verdict_d_structural.json
│   └── arm1_disposition.json
└── arm2_fits/
    ├── verdict_d_spectral_niche.json
    ├── verdict_d_structural_niche.json
    └── arm2_disposition.json

acoustic_features/
├── soundplot_features_v1.parquet     (1344 × 194)
├── birdnet_embeddings_v1.parquet     (1344 × 2052)
├── structural_features_v1.parquet    (1344 × 6)
└── dialect_clusters_v1.parquet       (1344 × 5)
```

## Methodology corpus integration

Substrate 7 of [[methodology_corpus_paper6_locked]] (the user's Paper 6 framework). The substrate inherits the discipline pattern from substrate 6 (gun violence):
- pre-registration via commit-hash timestamp
- cell-availability scoping before lock
- locked decision rules (≥5% + CI clean positive)
- parallel models for multi-dimensional outcomes
- honest negative reporting

The Phase 1 result here is parallel in structure to gun violence v0.6: a partial replication where one axis (spectral) is load-bearing and another (structural) is sub-threshold. That's the methodological discipline's signature — when you test multiple operationalizations, partial outcomes ARE the honest finding, not a bug to be hidden.

## Next steps (Phase 2 — not in this run)

1. **3-axis robustness on the spectral PASS**:
   - Source replication: pull Macaulay Library subset, re-fit, compare
   - Sample resampling: 5-fold cell-stratified random split, refit
   - Geographic refinement: swap BCR for L3 ecoregion partition, refit
2. **Structural reparameterization**: tighter half-normal priors on σ_cell, non-centered cell parameterization to clear divergences
3. **Industry-baseline comparison**: replicate Sainburg 2020 *PLOS Comp Bio* pipeline on our data; confirm our cell-method recovers what their unsupervised UMAP+HDBSCAN does
4. **Cross-species etymology pivot**: with dialect clusters per-species established, compare cluster structure across species (deferred to a Phase 2 substrate)
