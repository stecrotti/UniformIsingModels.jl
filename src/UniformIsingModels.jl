module UniformIsingModels

import OffsetArrays: OffsetVector, fill
import LinearAlgebra: dot
import UnPack: @unpack

export UniformIsing, energy, normalization, marginals!, marginals

include("accumulate.jl")
include("uniform_ising.jl")

end
