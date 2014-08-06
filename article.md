% Estimating the asset correlation $\rho$ on Dutch mortgage data
% P. van den Berg; A. Colangelo
% 6 August, 2014

## Introduction

[...]

## Model specification
The likelihood of finding $k_{bt}$ defaults out of $N_{bt}$
observations for subportfolios $b=1\dots B$, in periods $t=1\dots T$, given the value of the systemic factors $y_t$, the long-term average default rates $\lambda_b$ and the asset
correlation $\rho$, is given by:

$$P(k_{bt}|\rho,\lambda_b,y_t,N_{bt})=\left(\begin{matrix} k_t \\ N_t\end{matrix}\right) G_{bt}^{k_{bt}}
(1-G_{bt})^{N_{bt}-k_{bt}}$$ 

where the $G_{bt}$ are defined as

$$G_{bt}(y_t;\lambda_b,\rho)=\Phi(\frac{\Phi^{-1} (\lambda_b)-\sqrt{\rho} y_t}{\sqrt{1-\rho}}) $$

The systemic factors $\mathbf{y}$ are assumed to be independently normally distributed,

$$\mathbf{y}\sim\varphi(y_1)\dots\varphi(y_T)$$

For the sake of simplicity, we choose independent priors for the $\lambda_b$ and $\rho$. Since for $\rho=0$ a natural choice of priors for the $\lambda_b$ is

$$P(\lambda|\alpha,\beta,I) = \frac{1}{\mathrm{Beta}(\alpha,\beta)}\lambda^{\alpha-1}(1-\lambda)^{\beta-1}$$

we take this as the marginal prior for $\lambda$ as well. Here $\alpha=0.5,\beta=0.5$ corresponds to Jeffrey's prior and  $\alpha>1,\beta>1$ gives a prior centered on $\lambda=\frac{\alpha-1}{\beta-1}$. For $\rho$ we choose a uniform prior on $[0,1]$,

$$P(\rho) = \begin{matrix} 1 & 0<=\rho<=1 \\ 0 & \text{otherwise} \end{matrix}$$ 

Using Bayes’ theorem for inverting conditional probabilities, the joint posterior for
$\lambda$ and $\rho$ is then

$$P(\mathbf{\lambda},\rho,\mathbf{y}|\mathbf{k},\mathbf{N},I)\propto P(\mathbf{k}|\rho,\lambda,\mathbf{y},\mathbf{N},I)  P(\rho)\prod_{b} P(\lambda_b) \prod_t P(y_t)$$

The marginal posterior $P(\lambda,\rho|k,N,I)$ is found by integrating out the
unobserved variable $y$

We implement this model in Stan  [@stan-software:2014]. See the [appendix](#app) for the complete model code.

### Appendix: Stan implementation {#app}

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

      # real<lower=0>lambda_alpha[B];    # parameters of the beta distribution
      # real<lower=0>lambda_beta[B];     

    }
    transformed parameters {
      real<lower=0,upper=1> G[B,T];

      for (t in 1:T)
        for (b in 1:B)
          G[b,t] <- normal_cdf( (nu[b] - sqrt(rho) * y[t]) / sqrt(1-rho) , 0.0, 1.0);

    }
    model {
      for (b in 1:B)
        # normal_cdf(nu[b], 0.0, 1.0) ~ beta(0.5, 0.5);
        increment_log_prob(beta_log(normal_cdf(nu[b], 0.0, 1.0), 0.5, 0.5));
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
