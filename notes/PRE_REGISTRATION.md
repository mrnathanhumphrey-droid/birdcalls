# Bird Song Substrate — Pre-Registration (LOCKED)

**Lock date:** 2026-05-16
**Lock mechanism:** GitHub commit hash on push to `github.com/mrnathanhumphrey-droid/birdcalls`
**Substrate:** 7 of the Paper 6 methodology corpus
**Author:** Nathan Humphrey

This document supersedes `PRE_REGISTRATION_DRAFT.md`. All decisions below are immutable from the lock commit forward. Any post-lock deviation must be logged in `DEVIATION_LOG.md` with reason.

---

## §1 Substrate goal

Test the partial-pooling-of-residual-classes framework (Paper 6 corpus method) on bioacoustic data via two pre-registered arms:

- **Arm 1 — Dialect-geography residual structure.** Within-species, do `(species × BCR × dialect-cluster)` cells separately identify regional dialect residual variance beyond an additive (species + BCR) baseline?
- **Arm 2 — Acoustic-niche partition.** Within sympatric bird communities, do `(community × spectral-band × time-of-day)` cells separately identify acoustic-niche specialization beyond an additive (community + species) baseline?

Both arms set up a later cross-species etymology / cross-vocal-learning-system comparison (deferred to Phase 2 of this substrate).

---

## §2 Locked scoping decisions

### A. Pilot species (4)

- **White-crowned sparrow** (*Zonotrichia leucophrys*) — Marler classical dialect work, west-coast subspecies
- **Song sparrow** (*Melospiza melodia*) — wide range, ~52 subspecies, individual + regional variation
- **Carolina chickadee** (*Poecile carolinensis*) — Southeast US, fee-bee dialect zones
- **Marsh wren** (*Cistothorus palustris*) — well-documented eastern/western dialect divergence (east-west cross-check)

### B. Geographic frame

- **CONUS + adjacent North America.** Limited to xeno-canto recordings with `cnt:"United States"` filter at metadata pull stage.

### C. Region partition (Arm 1 cell axis)

- **USFWS / NABCI Bird Conservation Regions (BCR).** Shapefile sourced from USDA Forest Service Research Data Archive RDS-2021-0001, layer `BCR_NA.shp`. 36 BCRs across NA; 34 touched by our 4-species CONUS sample.

### D. Recording inclusion criteria

- `type` contains "song" (case-insensitive substring match) — excludes call-only recordings
- Has valid `lat`, `lon` coordinates
- Quality score `q` ∈ {"A", "B", "C"} — excludes "D" / "E" / "no score"
- After filters: 1,452 usable from 2,171 total CONUS recordings (67% retention)

### E. Cell-availability threshold

- **N ≥ 10 recordings per (species, BCR) cell** for inclusion as a primary cell in the dialect-cluster fit. Cells below threshold receive aggressive partial-pooling shrinkage but are not dropped.
- At lock: 40 (species, BCR) cells pass N ≥ 10. Per-species: WCSP 9, SOSP 16, CACH 5, MAWR 10.

### F. Feature layers (both LOCKED — analyzed in parallel)

**Spectral features (via SoundPlot + BirdNET):**
- BirdNET embedding-space distance (Cornell pre-trained model, version locked at install)
- MFCC-13 + delta-MFCC cosine distance via SoundPlot `FeatureExtractor.extract_all()`
- Spectral centroid + bandwidth + contrast + rolloff distribution KL-divergence

**Structural features (via SoundPlot syllable segmentation + custom metrics):**
- Syllable inventory size per recording (UMAP + HDBSCAN on onset-segmented syllables)
- Syllable-transition matrix Shannon entropy
- Song-bout sequence Levenshtein distance between recordings
- Lempel-Ziv compression ratio (structural complexity proxy)

### G. Model combination

- **Parallel models** (Option A). Fit Arm 1 twice — once with spectral distance outcome, once with structural distance outcome. Verdict requires BOTH to be CI-clean for a STRONG_REPLICATION on Arm 1. Discordance is informative and will be reported as-is.
- Same parallel-fit pattern for Arm 2 outcomes.

### H. Bayesian model spec

- Hierarchical Gaussian regression (continuous song-feature-distance outcomes), 4 chains × 1000 warmup + 1000 sampling, adapt_delta = 0.95, max_treedepth = 10.
- Cell-level random intercepts + random slopes on a per-cell continuous covariate (TBD per-arm).
- Same Stan-via-cmdstanpy pipeline as gun violence substrate.

### I. Arm 2 community definition

- **H3 hex level 5** (~252 km² hexagons) from xeno-canto recording lat/lon. Community = hex where ≥ 2 of our 4 pilot species have recordings.
- Community-level sympatry from our own recording footprint (avoids eBird API dependency).

### J. Arm 2 spectral-band partition

- **3 bands locked:** low (0–2 kHz), mid (2–6 kHz), high (6+ kHz).
- Time-of-day partition: dawn (04:00–08:00 local), day (08:00–18:00), dusk/night (18:00–04:00) — derived from xeno-canto `date` + `time` fields when available.

### K. Decision rules (LOCKED)

**Arm 1 — H_DIALECT_RESIDUAL:**
- PASS: cell partial-pooling adds **≥ 5% additional variance explained** vs additive (species + BCR) baseline on **BOTH** spectral AND structural outcomes, with 95% CrI clean positive on both.
- PARTIAL: PASS on one outcome (spectral or structural) but not the other.
- FAIL: Neither outcome passes the 5% + CI-clean threshold.

**Arm 1 — H_ADDITIVE_SUFFICIENT (null hypothesis):**
- Cell partial-pooling adds < 5% additional variance on either outcome; song-feature variation is captured by additive species + BCR effects alone.

**Arm 2 — H_NICHE_PARTITION:**
- PASS: cell partial-pooling adds **≥ 5% additional variance explained** vs additive (community + species) baseline on both spectral and structural call-overlap measures, with 95% CrI clean positive on both.
- PARTIAL / FAIL: as in Arm 1.

### L. 3-axis robustness pass (LOCKED, runs after main analysis)

Same pattern as gun violence substrate:
- **Source replication:** swap xeno-canto for Macaulay Library on a subset (≥ 20% of recordings, balanced across species). Compare coefficient.
- **Sample resampling:** 5-fold cell-stratified random split of recordings within each (species, BCR) cell, refit, compare. Pre-reg seed `20260616`.
- **Geographic refinement:** swap BCR for L3 ecoregion partition, refit, compare. (BCR is the primary lock; ecoregion is the refinement axis.)

### M. Industry-baseline comparison (LOCKED)

After main analysis lands, run two parallel comparisons:
- **BirdNET-embedding-only baseline:** UMAP + HDBSCAN on BirdNET embeddings within-species. Report whether unsupervised clustering recovers the same dialect-cluster structure as our cell-based partial-pooling model.
- **Sainburg et al. 2020 *PLOS Comp Bio* methodology baseline:** UMAP on spectral syllable features + syllable-transition graphs. Replicate their pipeline on our data and compare cluster structure.

The corpus-method contribution claim is conditional on the partial-pooling model surfacing residual structure that the unsupervised baselines miss.

---

## §3 Pre-registration discipline

- This document is LOCKED by commit hash on push to `github.com/mrnathanhumphrey-droid/birdcalls`.
- No outcome-level inspection (acoustic features, BirdNET embeddings, dialect-cluster output) has happened at lock time. Only metadata (count, type, quality, lat/lon, subspecies fields) has been inspected for cell-availability scoping.
- Any post-lock deviation logged in `DEVIATION_LOG.md` with the deviation, the reason, and the commit hash of the deviation.
- Robustness arms run with same discipline; their decision rules are locked here.

---

## §4 Methodology corpus integration

Substrate 7 of [[methodology_corpus_paper6_locked]]. Inherits the decoupled-cells design philosophy from substrate 6 (gun violence) plus the pre-reg + 3-axis-robustness + industry-baseline discipline pattern.

Related:
- [[gun_violence_state_2026_05_16]] — substrate 6 (Phase 1 wrapped, established the discipline pattern)
- [[methodology_corpus_paper6_locked]] — corpus framework
- [[corpus_qb_2026_05_16]] — current corpus state
