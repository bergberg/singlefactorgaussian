# Single factor one-period model

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

we take this as the marginal prior for $\lambda$ as well. Here $\alpha=0.5,\beta=0.5$ corresponds to Jeffrey's prior and  $\alpha>1,\beta>1$ gives a prior centered on $\lambda=\frac{\alpha-1}{\beta-1}$. For $rho$ we choose a uniform prior,

$$P(\rho) = \begin{matrix} 1 & 0<=\rho<=1 \\ 0 & \text{otherwise} \end{matrix}$$ 

Using Bayes’ theorem for inverting conditional probabilities, the joint posterior for
$\lambda$ and $\rho$ is then

$$P(\mathbf{\lambda},\rho,\mathbf{y}|\mathbf{k},\mathbf{N},I)\propto P(\mathbf{k}|\rho,\lambda,\mathbf{y},\mathbf{N},I)  P(\rho)\prod_{b} P(\lambda_b) \prod_t P(y_t)$$

The marginal posterior $P(\lambda,\rho|k,N,I)$ is found by integrating out the
unobserved variable $y$

We implement this model in Stan  [@stan-software:2014] as follows:

	