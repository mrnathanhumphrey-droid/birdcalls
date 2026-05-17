// Arm 1 — hierarchical Gaussian regression with cell partial-pooling.
// Outcome (y) = song-feature distance from species centroid (either spectral or structural).
// Cells: (species × BCR × dialect_cluster). Tests whether cell partial-pooling
// adds variance explained vs additive (species + BCR) baseline.
//
// We fit ONE multilevel model and compare:
//   Baseline component:  alpha_sp[species] + alpha_bcr[BCR]
//   Cell residual:       alpha_cell[cell]
// The fraction of total y-variance explained ONLY by cell residual
// (computed via posterior decomposition) is the test statistic.

data {
  int<lower=1> N;                       // recordings
  int<lower=1> S;                       // species
  int<lower=1> B;                       // BCRs
  int<lower=1> C;                       // cells (species × BCR × dialect_cluster)
  array[N] int<lower=1, upper=S> species;
  array[N] int<lower=1, upper=B> bcr;
  array[N] int<lower=1, upper=C> cell;
  vector[N] y;                          // continuous distance outcome
}

parameters {
  real mu;                              // global intercept
  vector[S] alpha_sp_raw;
  vector[B] alpha_bcr_raw;
  vector[C] alpha_cell_raw;
  real<lower=0> sigma_sp;
  real<lower=0> sigma_bcr;
  real<lower=0> sigma_cell;
  real<lower=0> sigma_y;
}

transformed parameters {
  vector[S] alpha_sp  = sigma_sp  * alpha_sp_raw;
  vector[B] alpha_bcr = sigma_bcr * alpha_bcr_raw;
  vector[C] alpha_cell= sigma_cell* alpha_cell_raw;
}

model {
  // Half-normal priors on scales (weakly informative on a z-scored y)
  sigma_sp  ~ normal(0, 1);
  sigma_bcr ~ normal(0, 1);
  sigma_cell~ normal(0, 1);
  sigma_y   ~ normal(0, 1);
  mu        ~ normal(0, 2);
  alpha_sp_raw  ~ std_normal();
  alpha_bcr_raw ~ std_normal();
  alpha_cell_raw~ std_normal();

  vector[N] eta;
  for (n in 1:N)
    eta[n] = mu + alpha_sp[species[n]] + alpha_bcr[bcr[n]] + alpha_cell[cell[n]];
  y ~ normal(eta, sigma_y);
}

generated quantities {
  // Variance components for the test statistic
  real var_sp   = variance(alpha_sp);
  real var_bcr  = variance(alpha_bcr);
  real var_cell = variance(alpha_cell);
  real var_resid= square(sigma_y);
  real var_total= var_sp + var_bcr + var_cell + var_resid;
  // Fraction of total variance attributable to the cell residual layer
  real frac_cell = var_cell / var_total;
  real frac_additive_baseline = (var_sp + var_bcr) / var_total;
}
