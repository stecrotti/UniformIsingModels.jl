# UniformIsingModels

[![Build Status](https://github.com/stecrotti/UniformIsingModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/UniformIsingModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/UniformIsingModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/stecrotti/UniformIsingModels.jl)

A fully-connected ferromagnetic [Ising model](https://en.wikipedia.org/wiki/Ising_model) with uniform coupling strength, described by a Boltzmann distribution

$p(\boldsymbol{\sigma}) = \frac{1}{Z} \exp\left[\beta\left(\frac{J}{N}\sum_{i<j}\sigma_i\sigma_j+\sum_{i=1}^Nh_i\sigma_i\right)\right],\quad \boldsymbol{\sigma}\in\\{-1,1\\}^N $

is exactly solvable in polynomial time.


| Quantity | Expression | Cost          |
| ------------- | ----------| ----------- |
| Normalization | $Z=\sum\limits_{\boldsymbol{\sigma}}\exp\left[\beta\left(\frac{J}{N}\sum_{i<j}\sigma_i\sigma_j+\sum_{i=1}^Nh_i\sigma_i\right)\right]$ | $\mathcal O (N^2)$ |
| Free energy | $F = -\frac{1}{\beta}\log Z$ | $\mathcal O (N^2)$ |
| Sample a configuration | $\boldsymbol{\sigma} \sim p(\boldsymbol{\sigma})$ | $\mathcal O (N^2)$ |
| Average energy | $U = \sum\limits_{\boldsymbol{\sigma}}p(\boldsymbol{\sigma})\left[-\left(\frac{J}{N}\sum_{i<j}\sigma_i\sigma_j+\sum_{i=1}^Nh_i\sigma_i\right)\right]$ | $\mathcal O (N^2)$ |
| Entropy | $S = -\sum\limits_{\boldsymbol{\sigma}}p(\boldsymbol{\sigma})\log p(\boldsymbol{\sigma})$ | $\mathcal O (N^2)$ |
| Distribution of the sum of the N spins | $p_S(s)=\sum\limits_{\boldsymbol{\sigma}}p(\boldsymbol{\sigma})\delta\left(s-\sum_{i=1}^N\sigma_i\right)$ | $\mathcal O (N^2)$ |
| Site magnetizations     | $m_i=\sum\limits_{\boldsymbol{\sigma}}p(\boldsymbol{\sigma})\sigma_i,\quad\forall i\in\{1,2,\ldots,N\}$ | $\mathcal O (N^3)$ |
| Correlations     | $r_{ij}=\sum\limits_{\boldsymbol{\sigma}}p(\boldsymbol{\sigma})\sigma_i\sigma_j,\quad\forall j\in\{1,2,\ldots,N\},i<j$ | $\mathcal O (N^5)$ |


## Example
```
]add UniformIsingModels
```
Construct a `UniformIsing` instance
```
using UniformIsingModels, Random

N = 10
J = 2.0
rng = MersenneTwister(0)
h = randn(rng, N)
β = 0.1
x = UniformIsing(N, J, h, β)
```
Compute stuff
```
# normalization and free energy
Z = normalization(x)
F = free_energy(x)

# energy and probability of a configuration
σ = rand(rng, (-1,1), N) 
E = energy(x, σ)
prob = pdf(x, σ)

# a sample along with its probability 
σ, p = sample(rng, x)

# single-site magnetizations <σᵢ>
m = site_magnetizations(x)

# distribution of the sum Σᵢσᵢ of all variables
ps = sum_distribution(x)

# energy expected value
U = avg_energy(x)

# entropy
S = entropy(x)

# correlations <σᵢσⱼ> and covariances <σᵢσⱼ>-<σᵢ><σⱼ>
p = correlations(x)
c = covariances(x)
```

## Notes
The internals rely on dynamic programming.

If you know of any implementation that's more efficient than this one I'd be very happy to learn about it!
