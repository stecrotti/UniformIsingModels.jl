struct UniformIsing{T<:Real, U<:OffsetVector}
    N :: Int                # number of spins
    J :: T                  # uniform coupling strength
    h :: Vector{T}          # external fields
    β :: T                  # inverse temperature
    L :: OffsetVector{U, Vector{U}}
    R :: OffsetVector{U, Vector{U}}
    Z :: T
    function UniformIsing(N::Int, J::T, h::Vector{T}; β = 1.0) where T
        @assert length(h) == N
        @assert J ≥ 0
        @assert β ≥ 0
        L = accumulate_left(h, β)
        R = accumulate_right(h, β)
        Z = sum( exp(β*J/2*(s^2/N-1))*L[N][s] for s in -N:N )
        new{T, eltype(L)}(N, J, h, β, L, R, Z)
    end
end

normalization(x::UniformIsing) = x.Z

pdf(x::UniformIsing, σ) = exp(-x.β*energy(x, σ)) / x.Z

function energy(x::UniformIsing, σ)
    s = sum(σ)
    f = dot(σ, x.h)
    -( x.J/2*(s^2/x.N-1) + f ) 
end

# p(σᵢ = +1)
function site_marginals!(p, x::UniformIsing)
    @unpack N, J, h, β, L, R, Z = x
    for i in eachindex(p)
        p[i] = 0
        for sL in -N:N
            for sR in -N:N
                p[i] += exp(β*J/2*((sL+1+sR)^2/N-1) + β*h[i]) * 
                            L[i-1][sL] * R[i+1][sR]
            end
        end
        p[i] /= Z
    end
    p
end
function site_marginals(x::UniformIsing{T,U}) where {T,U} 
    site_marginals!(zeros(T,x.N), x)
end

function site_magnetizations!(p, x::UniformIsing)
    @unpack N, J, h, β, L, R, Z = x
    for i in eachindex(p)
        p[i] = 0
        for sL in -N:N
            for sR in -N:N
                s = sL + sR
                p[i] += exp(β*J/2*((s^2+1)/N-1)) * 2sinh(β*(J*s/N + h[i])) * 
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
sample(x::UniformIsing) = sample!(zeros(Int, x.N), x)