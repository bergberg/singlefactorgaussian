data {
  int<lower=0> T; // time periods
  int<lower=0> B; // uniform risk categories (PD buckets)
  int k[B,T];
  int N[B,T];
}
parameters {
  real nu[B]; // nu = normal_cdf_inv(lambda)
  real<lower=0,upper=1> rho;
  real y[T];

  real<lower=0>lambda_alpha[B];    // parameters of the beta distribution
  real<lower=0>lambda_beta[B];     

}
transformed parameters {
  real<lower=0,upper=1> G[B,T];

  for (t in 1:T)
    for (b in 1:B)
      G[b,t] <- normal_cdf( (nu[b] - sqrt(rho) * y[t]) / sqrt(1-rho) , 0.0, 1.0);

}
model {

  
  # normal_cdf(nu, 0.0, 1.0) ~ beta(lambda_alpha,lambda_beta);
  for (b in 1:B)
    increment_log_prob(beta_log(normal_cdf(nu[b], 0.0, 1.0), lambda_alpha[b], lambda_beta[b]));


  rho ~ uniform(0.0,1.0);  
  y ~ normal(0.0,1.0);

  for (t in 1:T)
    for (b in 1:B)
      k[b,t] ~ binomial(N[b,t], G[b,t] );

}

generated quantities {
  real lambda[B];
  for (b in 1:B)
    lambda[b] <- normal_cdf(nu[b], 0.0, 1.0);
}
