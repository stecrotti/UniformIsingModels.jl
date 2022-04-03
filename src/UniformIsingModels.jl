module UniformIsingModels

import OffsetArrays: OffsetVector, fill
import LinearAlgebra: dot
import UnPack: @unpack
import Random: GLOBAL_RNG, AbstractRNG

export UniformIsing, energy, normalization, pdf, 
        site_magnetizations!, site_magnetizations,
        pair_magnetizations!, pair_magnetizations,
        sample!, sample

include("accumulate.jl")
include("uniform_ising.jl")

end
