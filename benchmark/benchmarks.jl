using BenchmarkTools
using UniformIsingModels

SUITE = BenchmarkGroup()

N = 100
J = 0.5
h = 1.2 * randn(N)
β = 2.3
x = UniformIsing(N, J, h, β)

SUITE["constructor"] = BenchmarkGroup()
SUITE["constructor"]["constructor"] = @benchmarkable UniformIsing($N, $J, $h, $β)

SUITE["observables"] = BenchmarkGroup()
SUITE["observables"]["normalization"] = @benchmarkable normalization(x)
SUITE["observables"]["entropy"] = @benchmarkable entropy(x)
SUITE["observables"]["site_magnetizations"] = @benchmarkable site_magnetizations(x)
# SUITE["observables"]["pair_magnetizations"] = @benchmarkable pair_magnetizations(x)
SUITE["observables"]["sum_distribution"] = @benchmarkable sum_distribution(x)