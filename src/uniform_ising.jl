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
                        # m[i,j] += 2*( exp(β*J/2*((s^2+4)/N-1))*sinh(β*(2J/N+h[i]+h[j])) -
                        #               exp(β*J/2*(s^2/N-1)) * sinh(β*(h[i]-h[j])) ) *
                        #               L[i-1][sL] * M[i+1,j-1][sM] * R[j+1][sR]
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