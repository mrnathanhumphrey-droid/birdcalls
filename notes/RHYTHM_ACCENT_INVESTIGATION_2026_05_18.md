# Rhythm-Accent Investigation — 2026-05-18

**Status:** NEGATIVE result. Hypothesis was worth investigating; data does not support it.

## Hypothesis

Birds have **rhythmic** (not tonal) accents, **affected by surrounding population**
(i.e., regional acoustic environment drives rhythm convergence across species).

This is distinct from the Phase 1/2 structural-feature null, which tested
*syntactic* features (syllable inventory, transition entropy, LZ complexity)
— rhythm specifically was never measured in those.

## What was tested

Song sparrow + marsh wren, both in 3 maximally-contrasted BCRs (COASTAL_CALIFORNIA,
SONORAN_AND_MOJAVE_DESERTS, NEW_ENGLAND/MID-ATLANTIC_COAST). 10 recordings per
(species × BCR) cell = 60 trajectories.

Rhythm-anchored 2D features extracted per frame:
- onset_strength (smoothed, librosa onset_strength, 25ms hop)
- ioi_rolling (inter-onset interval rolling mean over last 3 onsets, normalized)

Pairwise multivariate DTW (path-length normalized) on the 60 × 59 / 2 = 1770 pairs.
Categorized as: within-species same-BCR, within-species cross-BCR,
cross-species same-BCR, cross-species cross-BCR.

## Key test result

Hypothesis predicts: if rhythm is population-mediated, then
`DS_SB < WS_DB` — i.e., different species in the same region should have
SMALLER rhythm distances than the same species across regions.

| Pair type | n | median DTW |
|-----------|---|------------|
| WS_SB (same species, same BCR) | 270 | 0.0818 |
| WS_DB (same species, cross BCR) | 600 | 0.0829 |
| DS_SB (cross species, same BCR) | 300 | 0.0855 |
| DS_DB (cross species, cross BCR) | 600 | 0.0869 |

Mann-Whitney U (DS_SB < WS_DB, one-sided): **p = 0.91, r = −0.055.**

**The cross-species population-convergence claim is rejected at high power.**

## Secondary test: within-species regional accent

`WS_SB < WS_DB`? Same species, same BCR should be tighter than same species
across BCRs if regional accents exist.

- WS_SB median 0.0818 vs WS_DB median 0.0829
- p = 0.080, r = 0.059 — borderline trend, very small effect.

**Within-species regional rhythm differences exist as a tiny statistical
trend but are not detectable at conventional significance.**

## N=30 (preliminary) vs N=60 (replication)

At N=30 (5 per cell), the preliminary test suggested a clean ordering
supporting the hypothesis (WS_SB < DS_SB < DS_DB < WS_DB, p=0.074 borderline).

At N=60 (10 per cell), WS_DB dropped from 0.0909 to 0.0829. The N=30 finding
was sample noise driven by a few outlier pairs in the NE_COAST group. The
larger N revealed the true pattern: species genetics dominates rhythm.

## Verdict

| Sub-claim | Verdict |
|-----------|---------|
| Birds have within-species regional rhythm differences | WEAK TREND (p=0.08, tiny effect) |
| Rhythm is shaped by surrounding population (cross-species convergence) | REJECTED (p=0.91, hard null) |

## Why this was worth doing

The Phase 1/2 wrap concluded "structural confirmed null across 3 axes."
That null was on syntactic features, not rhythm. Rhythm is the specific
operationalization that hadn't been tested. Following the L1 cross-corpus
retro protocol, an untested operationalization is a Mode A candidate
worth probing before classifying as a substrate-level null.

The probe came back negative at higher power. Mode A path is now closed
for rhythm-anchored features as well: the corpus-level null on structural
features (broadly construed) is reinforced rather than overturned.

## What stays open

The Phase 1/2 substrate-7 main conclusions stand unchanged:
- Arm 1 spectral signal robust across 3 axes (+4.7pp over Sainburg-2020)
- Cell partial-pooling on BirdNET-embedding distance generalizes
- Structural/syntactic + rhythm operationalizations both null

This investigation just adds rhythm to the "tested operationalizations" list.
Sub-BCR geographic resolution + phrase-level rhythm-template features
remain untested but unlikely to be the next priority given the corpus-wide
pattern.

## Files

| Path | What |
|------|------|
| `_scripts/extract_rhythm_trajectories.py` | Song sparrow rhythm extraction |
| `_scripts/extract_marsh_wren_trajectories.py` | Marsh wren rhythm extraction |
| `_scripts/dtw_pairwise_rhythm_test.py` | N=15 within-BCR DTW test (suggestive) |
| `_scripts/dtw_cross_species_test.py` | N=60 cross-species DTW test (negative) |
| `_scripts/build_3d_umap.py` | 3D UMAP scatter for Blender viz |
| `_scripts/inline_3d_viz.py` | Self-contained Three.js HTML build |
| `acoustic_features/rhythm_trajectories_v1.json` | 30 song-sparrow trajectories |
| `acoustic_features/rhythm_trajectories_marsh_wren_v1.json` | 30 marsh-wren trajectories |
| `viz/dtw_results.{json,md}` | Within-BCR test outputs |
| `viz/dtw_cross_species_results.md` | Cross-species test outputs |
| `viz/birdsong_3d.html`, `birdsong_3d_inline.html` | Three.js 3D scatter |
| `viz/birdsong_3d_points*.json` | Per-recording UMAP3 coords |
