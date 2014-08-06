# Bayesian model benchmarking (working title)
## Motivation
## Setup/data description 

We are given a dataset of ratings 

$$\left[ x_{ij}\right]_{I_{ij}=1}\in\mathbb{R}, i=1\dots N,j=1\dots M, I_{ij} \in [0,1]$$

of $M$ counterparties by $N$ banks. The ratings $x$ are transformed to be numbers on the real line; for instance, if given as probability of default estimates $p\in[0,1]$, we transform these as $x=\Phi^{-1}(p)$. Not all counterparties are rated by all banks; we write this as an incidence matrix $I_{ij}$ where $I_{ij}=1$ if a rating by bank $i$ for counterparty $j$ exists.

## Model description
Our model is particularly simple:

$$ 
x_{ij|I_{ij}=1}  \sim  \mathrm {Normal} (q_{j} + \mu_{i},\tau_i^{-1} ) 
$$

Each rating $x_{.j}$ is an estimate of the (unknown)  'true' rating $q_j$ of that counterparty. We assume that each bank $i$ uses a model characterized by a bias $\mu_i$ and a precision $\tau_i = \sigma_i^{-2}$ which are the same for each rating of that bank, and we assume that the $x_{.j}$ are uncorrelated.
As it is, this model suffers from an M-way invariance;  under the simultaneous transformations

$$
	\begin{matrix} q_j \to q_j+a_{ij} \\ \mu_i\to\mu_i-a_{ij} \end{matrix}
$$

for any set of exclusive[^1] pairs $(i,j)$  the resulting distribution of $x_{ij}$ will be the same.
We can remove the collinearity by setting (for instance) all $q_j  = \sum_{i=1}^N I_{ij} x_{ij} / \sum_{i=1}^N I_{ij}, j = 1\dots M$[^2] and restate our model in terms of $x_{ij} \to x_{ij}-q_j$. The bias parameters then represent the bias relative to the average rating (which may, of course, itself be a biased estimate).Note that this also removes any dependence on $x_{.j}$ where $\sum_1^N x_{ij}=1$, i.e., counterparties for which only one rating is available. 
We wish to estimate the marginal posterior density for the $\mu_i$,
$$P(\mu|x,I,M,N)=\int\dots\int P(\mu|\tau,x,I,M,N)\mathrm{d}\tau_1 \dots \mathrm{d}\tau_M$$

We choose the usual conjugate joint priors for the $\mu_i$ and $\tau_i$,

$$\mathrm{P}(\mu_i,\tau_i|\dots)=\mathrm{NormalGamma}(\mu_{0i},\nu_i,\alpha_i,\beta_i)$$

with 
$$\begin{matrix} \mu_{0i}=0, i=1\dots N \\ \nu_i \end{matrix}$$
### Implementation in Stan
Stan [@stan-software:2014]

[^1]: I.e., choose all pairs $(i,j)$ such that each $i$ and $j$ occur at most once
[^2]: Alternatively, we could set some constraint on the $q_j$ by specifying a prior which breaks the symmetry.
