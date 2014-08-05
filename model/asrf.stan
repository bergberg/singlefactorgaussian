data {
  int<lower=0> T;
  int k[T];
  int N[T];
}
parameters {
  real nu; // nu = normal_cdf_inv(lambda)
  real<lower=0,upper=1> rho;
  real y[T];

  real<lower=0>lambda_alpha;    // parameters of the beta distribution
  real<lower=0>lambda_beta;     

}
transformed parameters {
  real<lower=0,upper=1> G[T];

  for (t in 1:T)
    G[t] <- normal_cdf( (nu - sqrt(rho) * y[t]) / sqrt(1-rho) , 0.0, 1.0);

}
model {

  
  # normal_cdf(nu, 0.0, 1.0) ~ beta(lambda_alpha,lambda_beta);

  increment_log_prob(beta_log(normal_cdf(nu, 0.0, 1.0), lambda_alpha, lambda_beta));


  rho ~ uniform(0.0,1.0);  
  y ~ normal(0.0,1.0);

  for (t in 1:T)
    k[t] ~ binomial(N[t], G[t] );

}
generated quantities {
  real lambda;
  lambda <- normal_cdf(nu, 0.0, 1.0);
}
