PD model calibration
====================

Single factor one-period model
------------------------------

The likelihood of finding $k$ defaults out of $N$
observations, given a long-term average default rate $\lambda$,  asset
correlation $\rho$, is given by:

$P(k|\rho,\lambda,y,N,I)=\frac{N!}{k!(N-k)!} G(y;\lambda,\rho)^{k}
(1-G(y;\lambda,\rho))^{N-k} \varphi(y)$

where

$G(y;\lambda,\rho)=\Phi(\frac{\Phi^{-1} (\lambda)-\sqrt{\rho} y}{\sqrt{1-\rho}}) $

and $y$ is the normally distributed single factor representing systemic risk. Using
Bayes’ theorem for inverting conditional probabilities, the posterior for
$\lambda$ is

$P(\lambda|\rho,y,k,N,I)=\frac{P(k|\rho,\lambda,y,N,I)P(\lambda|\rho,y,N,I)}{P(k|N,I)}$

*If we choose a beta prior for lambda:* $P(\lambda|\rho,y,N,\alpha,\beta,I)d\lambda=\frac{\lambda^{\alpha-1}(1-\lambda)^{\beta-1}}{\mathrm{Beta}(\alpha,\beta)}$


$P(\lambda|\rho,y,k,N,I)\propto G(y;\lambda,\rho)^{k}
(1-G(y;\lambda,\rho))^{N-k}\varphi(y)\lambda^{\alpha-1}(1-\lambda)^{\beta-1}$

The marginal posterior $P(\lambda|\rho,k,N,I)$ is found by integrating out the
unobserved variable $y$

$P(\lambda|\rho,k,N,I)\propto \int_0^1 (1-G(y;\lambda,\rho))^{N-k}\varphi(y)\lambda^{\alpha-1}(1-\lambda)^{\beta-1} \mathrm{d}y$

On the whole, it is preferable to sample from the posterior using Markov chain Monte-Carlo techniques. We implement this model in Stan as follows:

