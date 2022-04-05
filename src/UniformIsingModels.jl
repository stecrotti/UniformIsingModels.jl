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
        sum_distribution!, sum_distribution,
        sample!, sample,
        avg_energy, entropy, free_energy

include("accumulate.jl")

struct UniformIsing{T<:Real, U<:OffsetVector}
    N :: Int                # number of spins
    J :: T                  # uniform coupling strength
    h :: Vector{T}          # external fields
    β :: T                  # inverse temperature
    L :: OffsetVector{U, Vector{U}} # partial sums from the left
    R :: OffsetVector{U, Vector{U}} # partial sums from the right
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

free_energy(x::UniformIsing) = -1/x.β*log(x.Z)

function sample_spin(rng::AbstractRNG, p::Real)
    @assert 0 ≤ p ≤ 1
    r = rand(rng)
    r < p ? 1 : -1
end

# return a sample along with its probability
function sample!(rng::AbstractRNG, σ, x::UniformIsing)
    @unpack N, J, h, β, L, R, Z = x
    a = 0.0; b = 0
    f(s) = β*J/2*(s^2/N-1)
    for i in 1:N
        tmp_plus = tmp_minus = 0.0
        for s in -N:N
            tmp_plus += exp(f(b+1+s)) * R[i+1][s]
            tmp_minus += exp(f(b-1+s)) * R[i+1][s]
        end
        p_plus = exp(β*h[i]) * tmp_plus
        p_minus = exp(-β*h[i]) * tmp_minus
        p_i = p_plus / (p_plus + p_minus)
        σi = sample_spin(rng, p_i)
        σ[i] = σi
        a += h[i]*σi
        b += σi
    end
    p = exp(f(b) + β*a) / Z
    @assert a == dot(h, σ); @assert b == sum(σ)
    σ, p
end
sample!(σ, x::UniformIsing) = sample!(GLOBAL_RNG, σ, x)
sample(rng::AbstractRNG, x::UniformIsing) = sample!(rng, zeros(Int, x.N), x)
sample(x::UniformIsing) = sample(GLOBAL_RNG, x)

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

# distribution of the sum of all variables as an OffsetVector
function sum_distribution!(p, x::UniformIsing)
    @unpack N, J, β, L, Z = x
    for s in -N:N
        p[s] = exp(β*J/2*(s^2/N-1))*L[N][s] / Z 
    end
    p
end
function sum_distribution(x::UniformIsing{T,U}) where {T,U} 
    p = fill(zero(T), -x.N:x.N)
    sum_distribution!(p, x)
end

function avg_energy(x::UniformIsing{T}; 
    p = sum_distribution(x), m = site_magnetizations(x)) where T
    @unpack N, J, h, β, L, R, Z = x
    s2 = sum(s^2*p[s] for s in eachindex(p))
    U_pairs = J/2*(s2/N-1)
    U_sites = dot(h, m)
    - (U_pairs + U_sites)
end

entropy(x::UniformIsing; kw...) = x.β * (avg_energy(x; kw...) - free_energy(x)) 

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

end # end module