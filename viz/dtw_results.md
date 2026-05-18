# DTW Pairwise Rhythm-Trajectory Test

Run at: 2026-05-18 06:56:48

Test: within-BCR DTW distance < between-BCR DTW distance? Mann-Whitney U one-sided.


## Headline

- **Overall p = 0.166464**
- **Rank-biserial r = 0.122** (positive = within smaller, supports accent hypothesis)
- Within-BCR median: 0.0666 (n=30)
- Between-BCR median: 0.0787 (n=75)
- Difference: 0.0122

## Verdict

**Hypothesis not supported at this sample size (p >= 0.05).** Within-BCR vs between-BCR distance distributions don't separate cleanly. Possible: (a) need different rhythm features, (b) need finer geographic resolution than BCR, (c) sample too small.


## Per-BCR breakdown

| BCR | n_within | n_between | med_within | med_between | p | r |
|---|---|---|---|---|---|---|
| COASTAL_CALIFORNIA | 10 | 50 | 0.0699 | 0.0761 | 0.4175 | 0.044 |
| SONORAN_AND_MOJAVE_DESERTS | 10 | 50 | 0.0529 | 0.0733 | 0.0002 | 0.720 |
| NEW_ENGLAND/MID-ATLANTIC_COAST | 10 | 50 | 0.1740 | 0.0872 | 0.9858 | -0.440 |