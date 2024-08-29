mutable struct UniformIsing{T<:Real, U<:OffsetVector}
    J    :: T                  # uniform coupling strength
    h    :: Vector{T}          # external fields
    β    :: T                  # inverse temperature
    L    :: OffsetVector{U, Vector{U}} # partial sums from the left
    R    :: OffsetVector{U, Vector{U}} # partial sums from the right
    dLdβ ::  OffsetVector{U, Vector{U}} # Derivative of L wrt β

    function UniformIsing(N::Integer, J::T, h::Vector{T}, β::T=1.0) where T
        @assert length(h) == N
        @assert β ≥ 0
        R = accumulate_right(h, β)
        L, dLdβ = accumulate_d_left(h, β)
        U = eltype(L)
        return new{T, U}(J, h, β, L, R, dLdβ)
    end
end
function UniformIsing(N::Integer, J::T, β::T=1.0)  where {T<:Real}
    h = zeros(T, N)
    return UniformIsing(N, J, h, β)
end

# re-compute the partial quantities needed to compute observables, in case some parameter (`J,h,β`) was modified
function recompute_partials!(x::UniformIsing)
    (; h, β, L, R, dLdβ) = x
    accumulate_left!(L, h, β)
    accumulate_right!(R, h, β)
    accumulate_d_left!(L, dLdβ, h, β)
end

nvariables(x::UniformIsing) = length(x.h)
variables(x::UniformIsing) = 1:nvariables(x)

function energy(x::UniformIsing, σ)
    s = sum(σ)
    f = dot(σ, x.h)
    N = nvariables(x)
    return -( x.J/2*(s^2/N-1) + f ) 
end

function lognormalization(x::UniformIsing)
    (; β, J, L) = x
    N = nvariables(x)
    return logsumexp( β*J/2*(s^2/N-1) + L[N][s] for s in -N:N )
end
function normalization(x::UniformIsing; logZ = lognormalization(x))
    return exp(logZ)
end

pdf(x::UniformIsing, σ) = exp(-x.β*energy(x, σ) - lognormalization(x))

free_energy(x::UniformIsing; logZ = lognormalization(x)) = -logZ / x.β

function sample_spin(rng::AbstractRNG, p::Real)
    @assert 0 ≤ p ≤ 1
    r = rand(rng)
    return r < p ? 1 : -1
end

# return a sample along with its probability
function sample!(rng::AbstractRNG, σ, x::UniformIsing; logZ = lognormalization(x))
    (; J, h, β, R) = x
    N = nvariables(x)
    a = 0.0; b = 0
    f(s) = β*J/2*(s^2/N-1)
    for i in 1:N
        tmp_plus = tmp_minus = 0.0
        for s in -N:N
            tmp_plus += exp(f(b+1+s) + R[i+1][s])
            tmp_minus += exp(f(b-1+s) + R[i+1][s])
        end
        p_plus = exp(β*h[i]) * tmp_plus
        p_minus = exp(-β*h[i]) * tmp_minus
        p_i = p_plus / (p_plus + p_minus)
        σi = sample_spin(rng, p_i)
        σ[i] = σi
        a += h[i]*σi
        b += σi
    end
    p = exp(f(b) + β*a - logZ)
    @assert a == dot(h, σ); @assert b == sum(σ)
    return σ, p
end
sample!(σ, x::UniformIsing; kw...) = sample!(default_rng(), σ, x; kw...)
sample(rng::AbstractRNG, x::UniformIsing; kw...) = sample!(rng, zeros(Int, nvariables(x)), x; kw...)
sample(x::UniformIsing; kw...) = sample(default_rng(), x; kw...)

# first store in `p[i]` the quantity log(p(σᵢ=+1)), then transform at the end 
function site_magnetizations!(p, x::UniformIsing; logZ = lognormalization(x))
    (; J, h, β, L, R) = x
    N = nvariables(x)
    f(s) = β*J/2*(s^2/N-1)
    for i in eachindex(p)
        p[i] = -Inf
        for sL in -N:N
            for sR in -N:N
                s = sL + sR
                p[i] = logaddexp(p[i], f(s+1) + β*h[i] + L[i-1][sL] + R[i+1][sR])
            end
        end
        # include normalization
        p[i] = exp(p[i] - logZ)
        # transform form p(+) to m=2p(+)-1
        p[i] = 2*p[i] - 1
    end
    return p
end
function site_magnetizations(x::UniformIsing{T,U}; kw...) where {T,U} 
    return site_magnetizations!(zeros(T, nvariables(x)), x; kw...)
end

# distribution of the sum of all variables as an OffsetVector
function sum_distribution!(p, x::UniformIsing; logZ = lognormalization(x))
    (; J, β, L) = x
    N = nvariables(x)
    for s in -N:N
        p[s] = exp( β*J/2*(s^2/N-1) + L[N][s] - logZ ) 
    end
    return p
end
function sum_distribution(x::UniformIsing{T,U}; kw...) where {T,U} 
    p = fill(zero(T), -nvariables(x):nvariables(x))
    return sum_distribution!(p, x; kw...)
end

function avg_energy(x::UniformIsing{T}; logZ = lognormalization(x)) where T
    (; J, β, L, dLdβ) = x
    N = nvariables(x)
    Zt = sum( exp( β*J/2*(s^2/N-1) + L[N][s]) * (J/2*(s^2/N-1)+dLdβ[N][s]) for s in -N:N)
    return -exp(log(Zt) - logZ)
end

entropy(x::UniformIsing; kw...) = x.β * (avg_energy(x; kw...) - free_energy(x; kw...)) 

function pair_magnetizations!(m, x::UniformIsing{T,U};
        M = accumulate_middle(x.h, x.β), logZ = lognormalization(x)) where {T,U}
    (; J, h, β, L, R) = x
    N = nvariables(x)
    Z = exp(logZ)
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
                            exp(L[i-1][sL] + R[j+1][sR]) 
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
                                  exp(L[i-1][sL] + M[i+1,j-1][sM] + R[j+1][sR])
                    end 
                end
            end
            m[i,j] /= Z; m[j,i] = m[i,j]
        end
    end
    return m
end
function pair_magnetizations(x::UniformIsing{T,U}; kw...) where {T,U} 
    pair_magnetizations!(zeros(T,nvariables(x),nvariables(x)), x; kw...)
end

function correlations!(c, x::UniformIsing; logZ = lognormalization(x),
        m = site_magnetizations(x; logZ), p = pair_magnetizations(x; logZ))
    N = nvariables(x)
    for i in 1:N
        for j in 1:N
            c[i,j] = p[i,j] - m[i]*m[j]
        end
    end
    return c
end
function correlations(x::UniformIsing{T,U}; kw...) where {T,U} 
    correlations!(zeros(T,nvariables(x),nvariables(x)), x; kw...)
end