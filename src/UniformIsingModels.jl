module UniformIsingModels

import OffsetArrays: OffsetVector, fill
import LinearAlgebra: dot
import Base: show
import UnPack: @unpack
import Random: GLOBAL_RNG, AbstractRNG

export UniformIsing, energy, normalization, pdf, 
        site_magnetizations!, site_magnetizations,
        pair_magnetizations!, pair_magnetizations,
        correlations!, correlations,
        sample!, sample,
        avg_energy, entropy, free_energy

include("accumulate.jl")

struct UniformIsing{T<:Real, U<:OffsetVector}
    N :: Int                # number of spins
    J :: T                  # uniform coupling strength
    h :: Vector{T}          # external fields
    β :: T                  # inverse temperature
    L :: OffsetVector{U, Vector{U}} # pre-computed useful quantities
    R :: OffsetVector{U, Vector{U}} # pre-computed useful quantities
    Z :: T                  # normalization
    function UniformIsing(N::Int, J::T, h::Vector{T}, β::T=1.0) where T
        @assert length(h) == N
        @assert J ≥ 0
        @assert β ≥ 0
        L = accumulate_left(h, β)
        R = accumulate_right(h, β)
        Z = sum( exp(β*J/2*(s^2/N-1))*L[N][s] for s in -N:N )
        new{T, eltype(L)}(N, J, h, β, L, R, Z)
    end
end

function show(io::IO, x::UniformIsing)
    @unpack N, J, h, β = x
    println(io, "UniformIsing with N = $N variables at temperature β = $β")
end

function energy(x::UniformIsing, σ)
    s = sum(σ)
    f = dot(σ, x.h)
    -( x.J/2*(s^2/x.N-1) + f ) 
end

pdf(x::UniformIsing, σ) = exp(-x.β*energy(x, σ)) / x.Z

function site_magnetizations!(p, x::UniformIsing)
    @unpack N, J, h, β, L, R, Z = x
    f(s) = β*J/2*(s^2/N-1)
    for i in eachindex(p)
        p[i] = 0
        for sL in -N:N
            for sR in -N:N
                s = sL + sR
                p[i] += ( exp( f(s+1) + β*h[i] ) - exp( f(s-1) - β*h[i] ) ) *
                            L[i-1][sL] * R[i+1][sR]
            end
        end
        p[i] /= Z
    end
    p
end
function site_magnetizations(x::UniformIsing{T,U}) where {T,U} 
    site_magnetizations!(zeros(T,x.N), x)
end

function pair_magnetizations!(m, x::UniformIsing{T,U};
        M = accumulate_middle(x.h, x.β)) where {T,U}
    @unpack N, J, h, β, L, R, Z = x
    f(s) = β*J/2*(s^2/N-1)
    for i in 1:N
        # j = i
        m[i,i] = 1
        # j = i+1
        j = i + 1
        j > N && break
        m[i,j] = 0
        for sL in -N:N
            for sR in -N:N
                s = sL + sR
                m[i,j] += ( exp( f(s+2) + β*(h[i]+h[j]) ) + 
                            exp( f(s-2) - β*(h[i]+h[j]) ) -
                            exp( f(s)   + β*(h[i]-h[j]) ) - 
                            exp( f(s)   - β*(h[i]-h[j]) )   ) * 
                            L[i-1][sL] * R[j+1][sR]
            end
        end
        m[i,j] /= Z; m[j,i] = m[i,j]
        # j > i + 1
        for j in i+2:N
            m[i,j] = 0
            for sM in -N:N
                for sL in -N:N
                    for sR in -N:N
                        s = sL + sM + sR 
                        m[i,j] += ( exp( f(s+2) + β*(h[i]+h[j]) ) + 
                                    exp( f(s-2) - β*(h[i]+h[j]) ) -
                                    exp( f(s)   + β*(h[i]-h[j]) ) - 
                                    exp( f(s)   - β*(h[i]-h[j]) )   ) *
                                  L[i-1][sL] * M[i+1,j-1][sM] * R[j+1][sR]
                    end
                end
            end
            m[i,j] /= Z; m[j,i] = m[i,j]
        end
    end
    m
end
function pair_magnetizations(x::UniformIsing{T,U}; kw...) where {T,U} 
    pair_magnetizations!(zeros(T,x.N,x.N), x; kw...)
end

function correlations!(c, x::UniformIsing;
        m = site_magnetizations(x), p = pair_magnetizations(x))
    N = x.N
    for i in 1:N
        for j in 1:N
            c[i,j] = p[i,j] - m[i]*m[j]
        end
    end
    c
end
function correlations(x::UniformIsing{T,U}; kw...) where {T,U} 
    correlations!(zeros(T,x.N,x.N), x; kw...)
end

function sample_spin(rng::AbstractRNG, p::Real)
    @assert 0 ≤ p ≤ 1
    r = rand(rng)
    r < p ? 1 : -1
end

# hierarchical sampling
# return a sample along with its probability
function sample!(rng::AbstractRNG, σ, x::UniformIsing)
    @unpack N, J, h, β, L, R, Z = x
    a = 0.0; b = 0
    for i in 1:N
        tmp = 0.0
        for s in -N:N
            tmp += exp(β*J/2*((b+1+s)^2/N-1)) * R[i+1][s]
        end
        pi = exp(β*(a + h[i])) / Z * tmp
        σi = sample_spin(rng, pi)
        σ[i] = σi
        a += h[i]*σi
        b += σi
    end
    p = exp(β*(J/2*(b^2/N-1) + a)) / Z
    σ, p
end
sample!(σ, x::UniformIsing) = sample!(GLOBAL_RNG, σ, x)
sample(rng::AbstractRNG, x::UniformIsing) = sample!(rng, zeros(Int, x.N), x)
sample(x::UniformIsing) = sample(GLOBAL_RNG, x)

function avg_energy(x::UniformIsing; 
        m = site_magnetizations(x), p = pair_magnetizations(x))
    E_sites = E_pairs = 0.0
    for i in 1:x.N
        E_sites += m[i]*x.h[i]
        for j in i+1:x.N
            E_pairs += p[i,j]
        end
    end
    E_pairs *= x.J/x.N
    - (E_sites + E_pairs)
end

free_energy(x::UniformIsing) = -1/x.β*log(x.Z)
entropy(x::UniformIsing; kw...) = x.β * (avg_energy(x; kw...) - free_energy(x)) 

end # end module