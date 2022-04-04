# UniformIsingModels

[![Build Status](https://github.com/stecrotti/UniformIsingModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/UniformIsingModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/UniformIsingModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/stecrotti/UniformIsingModels.jl)

A fully-connected ferromagnetic Ising model with uniform coupling strength, described by a Boltzmann distribution

>![equation](https://latex.codecogs.com/svg.image?p(\sigma|J,&space;\boldsymbol{h},&space;\beta)&space;=&space;\frac{1}{Z_{J,&space;\boldsymbol{h},&space;\beta}}\exp\left[\beta\left(\frac{J}{N}\sum_{i<j}\sigma_i\sigma_j&space;&plus;\sum_{i=1}^Nh_i\sigma_i\right)\right])

is exactly solvable in polynomial time.


| Quantity | Cost          |
| ------------- | ----------- |
| Normalization, Free energy      |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N^2)) |
| Sample a configuration      |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N^2)) |
| Site magnetizations     | ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N^3))     |
| Pair magnetizations, Correlations |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N^5))  |
| Average energy, Entropy |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N^5))  |

## Example

Construct a `UniformIsing` instance
```
using UniformIsingModels, Random

N = 10
J = 2.0
h = randn(MersenneTwister(0), 10)
β = 0.1
x = UniformIsing(N, J, h; β=β)
```
Compute stuff
```
# normalization
Z = x.Z

# magnetizations
m = site_magnetizations(x)

# pairwise correlations
c = correlations(x)

# a sample along with its probability 
σ, p = sample(MersenneTwister(0), x)

# energy expected value
U = avg_energy(x)
```
