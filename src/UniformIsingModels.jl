module UniformIsingModels

import OffsetArrays: OffsetVector, fill
import LinearAlgebra: dot
import UnPack: @unpack
import Random: GLOBAL_RNG, AbstractRNG

export UniformIsing, energy, normalization, pdf, marginals!, marginals,
        sample!, sample

include("accumulate.jl")
include("uniform_ising.jl")

end
