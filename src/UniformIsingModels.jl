module UniformIsingModels

import OffsetArrays: OffsetVector, fill
import LinearAlgebra: dot
import Base: show
import UnPack: @unpack
import Random: GLOBAL_RNG, AbstractRNG
import LogExpFunctions: logsumexp, logaddexp, logsubexp

export UniformIsing, energy, normalization, pdf, 
        site_magnetizations!, site_magnetizations,
        pair_magnetizations!, pair_magnetizations,
        correlations!, correlations,
        sum_distribution!, sum_distribution,
        sample!, sample,
        avg_energy, entropy, free_energy

include("accumulate.jl")
include("uniform_ising.jl")