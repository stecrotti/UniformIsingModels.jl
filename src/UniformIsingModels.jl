module UniformIsingModels

using OffsetArrays: OffsetVector, fill
using LinearAlgebra: dot
using Random: default_rng, AbstractRNG
using LogExpFunctions: logsumexp, logaddexp

export UniformIsing, nvariables, variables, recompute_partials!,
        energy, lognormalization, normalization, pdf,
        avg_energy, entropy, free_energy, 
        site_magnetizations!, site_magnetizations,
        pair_magnetizations!, pair_magnetizations,
        correlations!, correlations,
        sum_distribution!, sum_distribution,
        sample!, sample

include("accumulate.jl")
include("uniform_ising.jl")

end # end module