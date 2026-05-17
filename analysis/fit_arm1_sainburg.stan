// Sainburg-2020-style baseline: outcome predicted by dialect_cluster ALONE.
// No species, no BCR layers. This is the "unsupervised UMAP+HDBSCAN clusters as
// the only predictor" approximation of Sainburg et al. 2020 *PLOS Comp Bio*.
// We compare frac_dc against our full-model frac_cell.

data {
  int<lower=1> N;
  int<lower=1> D;                       // distinct dialect_cluster labels
  array[N] int<lower=1, upper=D> dc;
  vector[N] y;
}

parameters {
  real mu;
  vector[D] alpha_dc_raw;
  real<lower=0> sigma_dc;
  real<lower=0> sigma_y;
}

transformed parameters {
  vector[D] alpha_dc = sigma_dc * alpha_dc_raw;
}

model {
  sigma_dc ~ normal(0, 1);
  sigma_y  ~ normal(0, 1);
  mu       ~ normal(0, 2);
  alpha_dc_raw ~ std_normal();
  vector[N] eta;
  for (n in 1:N) eta[n] = mu + alpha_dc[dc[n]];
  y ~ normal(eta, sigma_y);
}

generated quantities {
  real var_dc = variance(alpha_dc);
  real var_y  = square(sigma_y);
  real frac_dc = var_dc / (var_dc + var_y);
}
