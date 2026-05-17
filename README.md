# Bird Song Substrate — Paper 6 Methodology Corpus

**Working title:** Cross-Species Acoustic-Niche Partition and Dialect-Geography Residual Structure in North American Birds

**Author:** Nathan Humphrey

**Status:** Pre-registration LOCKED at commit `aa38aea` (2026-05-16). Phase 1 PARTIAL_REPLICATION on both arms — spectral PASS, structural FAIL. **Phase 2 (2026-05-17): spectral PASS survives 2/3 robustness axes + outperforms Sainburg-2020 unsupervised baseline by +4.7 pp.**

**Phase 1 headline:** Cell partial-pooling on BirdNET-embedding distance recovers **19% of variance** in dialect-geography (Arm 1) and **21%** in acoustic-niche-partition (Arm 2). Structural/syntactic outcomes only recover 3-4% — below pre-reg 5% threshold. Bioacoustic differentiation operates primarily on spectral position (which syllables), not syntactic structure (how syllables are combined).

**Phase 2 robustness on Arm 1 spectral (3/3 axes PASS):**
- Sample resampling (5-fold cell-stratified, seed 20260616): **5/5 folds PASS** (range 16.0-19.7%)
- Geographic refinement (BCR → EPA L3 ecoregion, 55 ecoregions): **PASS, frac_cell = 0.217** (slightly stronger than BCR)
- Source replication, recordist-source split (substitute for Macaulay swap, see `notes/DEVIATION_LOG.md` Entry 001): **TOP recordists 0.265 PASS / LONGTAIL recordists 0.152 PASS** — signal is not a recordist-equipment artifact
- Source replication, canonical Macaulay swap: still deferred pending Cornell Lab research access

**Sainburg-2020 industry-baseline head-to-head:** Our partial-pooling (sp+BCR+cell) extracts **+4.7 pp more spectral variance and +6.5 pp more structural variance** than unsupervised UMAP+HDBSCAN clusters alone. The cell structure carries information beyond what the standard unsupervised pipeline finds.

**Geographic frame:** CONUS + North America (continental-USA breeding species and migrants, plus contiguous Canadian/Mexican range overlap where relevant for cells).

## Substrate goal

Substrate 7 of the Paper 6 methodology corpus. Tests the partial-pooling-of-residual-classes framework on bioacoustic data via two pre-registered arms:

### Arm 1 — Dialect-geography residual structure

**Question:** Does song-feature variance pool across geographic regions within species in a way that an additive (species + region) model cannot capture? Specifically, do (species × geographic-region × dialect-cluster) cells reveal a residual class that's separately identifiable from main effects?

**Cells (preliminary):** `(species × geographic-region × dialect-cluster)`. Cell definition will be locked in pre-reg.

**Outcome (preliminary):** song-feature distance from species-population centroid. Possible measures: syllable inventory divergence (Levenshtein on syllable sequence), spectral-centroid distribution, song-bout structure.

**Reference dialects in NA literature:** White-crowned sparrow (Marler), Song sparrow, Bewick's wren, Indigo bunting, Carolina chickadee. These are the pilot-species candidates.

### Arm 2 — Acoustic-niche partition

**Question:** Within sympatric bird communities, do species partition the acoustic niche (spectral band × time-of-day × call type) more than would be predicted by their taxonomic identity alone? Does a cell-partition residual recover acoustic-niche specialization as a separately-identifiable class?

**Cells (preliminary):** `(community × spectral-band × time-of-day)`. Communities defined by eBird co-occurrence at a geography/time resolution to be locked.

**Outcome (preliminary):** call-overlap or call-interference rate between sympatric species, measured from acoustic features. Specifics in pre-reg.

### Why both arms

The two arms set up a later **cross-species etymology pivot**: if dialect-geography residual structure replicates across species, AND acoustic-niche partitioning is a separately-identifiable residual class, then the corpus methodology has a defensible cross-vocal-learning-system claim independent of human linguistics. The dialect-arm results are then directly comparable to human-dialect work in a future paired-substrate test.

## Data sources

- **Xeno-canto** (https://xeno-canto.org/) — open citizen-science archive, full API, CC-BY-SA / CC-BY-NC metadata
- **Macaulay Library** (https://www.macaulaylibrary.org/) — Cornell Lab academic archive, API key required, higher-quality recordings
- **BirdNET** (https://birdnet.cornell.edu/) — Cornell's pre-trained acoustic classifier, feature extraction
- **eBird** (https://ebird.org/) — observation/range data for cell community definition
- **3D birdsong modeling repo (TBD)** — user-supplied open-source bioacoustic modeling git; will be linked here once received

## Discipline

- All decision rules and cell operationalizations LOCKED in `notes/PRE_REGISTRATION.md` via GitHub commit hash BEFORE any acoustic-feature or outcome analysis
- Cell-availability scoping (metadata-only, no outcome inspection) precedes pre-reg lock
- Public commit chain (gun-violence model)
- Mid-tier-publishable first pass; cross-species etymology pivot deferred to phase 2

## Methodology corpus context

This study is substrate 7 of a locked methodology corpus (Paper 6) demonstrating the partial-pooling-of-residual-classes framework across heterogeneous domains:

1. Collatz tail-behavior modeling
2. NBA Projections (residual-class offsets, BLK × Center coupling)
3. SP500 a_final residue classes
4. CancerResearch Paper 1+2 (Lock 2022 spike-and-slab pan-cancer survival)
5. CancerResearch Paper 3 — MOFA-FLEX niche joint refit (FALSIFIED, substrate boundary)
6. Gun violence cross-geography (Phase 1 wrapped, externally gated for tract-level upgrade)
7. **Bird song dialect-geography + acoustic-niche partition (this study)**

The decoupled-cells design from substrate 6 carries forward conceptually: cells defined where the variables of interest vary independently across the sampling frame.
