# Single factor one-period model

## Introduction

[...]

## Model specification
The likelihood of finding $k_t$ defaults out of $N_t$
observations in periods $t=1\dots T$, given the value of the systemic factors $y_t$, the long-term average default rate $\lambda$ and the asset
correlation $\rho$, is given by:

$$P(\mathbf{k}|\rho,\lambda,y,N_t,I)=\prod_{t=1}^{T}\left(\begin{matrix} k_t \\ N_t\end{matrix}\right) G_t^{k_t}
(1-G_t)^{N_t-k_t}$$ 

where the *latent variables* $G_t$ are defined as

$$G_t(y_t;\lambda,\rho)=\Phi(\frac{\Phi^{-1} (\lambda)-\sqrt{\rho} y_t}{\sqrt{1-\rho}}) $$

The systemic factors $\mathbf{y}$ are assumed to be independently normally distributed,

$$\mathbf{y}\sim\varphi(y_1)\dots\varphi(y_T)$$


We would like to choose conjugate priors for $\lambda$ and $\rho$. We can derive these in two steps by treating the model as a "hierarchical" model for $\mathbf{G}$. The priors for $\mathbf{G}$ should be the usual independent conjugate prior for parameters in a binomial distribution, with hyperparameters $\alpha$ and $\beta$:

$$P(\lambda,\rho,\mathbf{y}|\alpha,\beta,I) = \prod_{t=1}^{T}P(G_t|\alpha,\beta,I)d\lambda= \frac{1}{\mathrm{Beta}(\alpha,\beta)}\prod_{t=1}^{T}G_t^{\alpha-1}(1-G_t)^{\beta-1}$$

where $\alpha=0.5,\beta=0.5$ corresponds to Jeffrey's prior and  $\alpha>1,\beta>1$ gives a prior centered on $\lambda=\frac{\alpha-1}{\beta-1}$.
The conjugate prior for $\rho$




Using Bayes’ theorem for inverting conditional probabilities, the joint posterior for
$\lambda$ and $\rho$ is

$$P(\lambda,\rho|\mathbf{y},\mathbf{k},\mathbf{N},I)=\frac{P(\mathbf{k}|\rho,\lambda,\mathbf{y},\mathbf{N},I)P(\lambda,\rho|\mathbf{y},\mathbf{N},I)}{P(\mathbf{k}|\mathbf{N},I)}$$

The marginal posterior $P(\lambda,\rho|k,N,I)$ is found by integrating out the
unobserved variable $y$

We implement this model in Stan  [@stan-software:2014] as follows: