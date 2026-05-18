# Cross-Species DTW Test — Population-Mediated Accent

Run at: 2026-05-18 08:15:30


## Sample

- 15 song sparrows + 15 marsh wrens, both spanning 3 BCRs (5 per BCR per species)
- 1770 pairwise DTW distances on rhythm-anchored trajectories

## Category medians

| Category | label | n | median | mean | std |
|---|---|---|---|---|---|
| WS_SB | WITHIN-species + SAME-BCR (e.g., SS-CA ↔ SS-CA) | 270 | 0.0818 | 0.0871 | 0.0318 |
| WS_DB | WITHIN-species + CROSS-BCR (e.g., SS-CA ↔ SS-NE) | 600 | 0.0829 | 0.0895 | 0.0314 |
| DS_SB | CROSS-species + SAME-BCR (e.g., SS-CA ↔ MW-CA) | 300 | 0.0855 | 0.0909 | 0.0282 |
| DS_DB | CROSS-species + CROSS-BCR (e.g., SS-CA ↔ MW-NE) | 600 | 0.0869 | 0.0920 | 0.0308 |

## Key test

Hypothesis: if accents are population-mediated, then CROSS-species + SAME-BCR < WITHIN-species + CROSS-BCR.
- Cross-species same-BCR median: 0.0855 (n=300)
- Within-species cross-BCR median: 0.0829 (n=600)
- Mann-Whitney p=0.909713, rank-biserial r=-0.055

**VERDICT:** REJECTED — species-genetic rhythm dominates over regional convergence. Same species across BCRs are still closer than different species in the same BCR.
