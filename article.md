PD model calibration
====================

Single factor one-period model
------------------------------
%Model description

In this model, the likelihood of finding $k_t$ defaults out of $N_t$
observations in period $t$, given the value of the systemic factor $y$, the long-term average default rate $\lambda$ and the asset
correlation $\rho$, is given by:

$$P(k_t|\rho,\lambda,y,N_t,I)=\left(\begin{matrix} k_t \\ N_t\end{matrix}\right) G(y;\lambda,\rho)^{k_t}
(1-G(y;\lambda,\rho))^{N_t-k_t}$$

where

$$G(y;\lambda,\rho)=\Phi(\frac{\Phi^{-1} (\lambda)-\sqrt{\rho} y}{\sqrt{1-\rho}}) $$

The systemic factor $y$ is assumed to be normally distributed. Using
Bayes’ theorem for inverting conditional probabilities, the posterior for
$\lambda$ is

$$P(\lambda|\rho,y,k,N,I)=\frac{P(k|\rho,\lambda,y,N,I)P(\lambda|\rho,y,N,I)}{P(k|N,I)}$$

If we choose a beta prior for lambda, $$P(\lambda|\rho,y,N,\alpha,\beta,I)d\lambda=\frac{\lambda^{\alpha-1}(1-\lambda)^{\beta-1}}{\mathrm{Beta}(\alpha,\beta)}$$

the posterior for $\lambda$ is proportional to 

$$P(\lambda|\rho,y,k,N,I)\propto G(y;\lambda,\rho)^{k}
(1-G(y;\lambda,\rho))^{N-k}\varphi(y)\lambda^{\alpha-1}(1-\lambda)^{\beta-1}$$

The marginal posterior $P(\lambda|\rho,k,N,I)$ is found by integrating out the
unobserved variable $y$

$P(\lambda|\rho,k,N,I)\propto \int_0^1 (1-G(y;\lambda,\rho))^{N-k}\varphi(y)\lambda^{\alpha-1}(1-\lambda)^{\beta-1} \mathrm{d}y$

We implement this model in Stan  [@stan-software:2014] as follows: