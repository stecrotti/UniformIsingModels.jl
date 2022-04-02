struct UniformIsing{T<:Real, U<:OffsetVector}
    N :: Int                # number of spins
    J :: T                  # uniform coupling strength
    h :: Vector{T}          # external fields
    β :: T                  # inverse temperature
    L :: OffsetVector{U, Vector{U}}
    R :: OffsetVector{U, Vector{U}}
    function UniformIsing(N::Int, J::T, h::Vector{T}; β = 1.0) where T
        @assert length(h) == N
        @assert J ≥ 0
        @assert β ≥ 0
        L = accumulate_left(h, β)
        R = accumulate_right(h, β)
        new{T, eltype(L)}(N, J, h, β, L, R)
    end
end

function normalization(x::UniformIsing)
    N = x.N
    sum( exp(x.β*x.J/2*(s^2/N-1))* x.L[N][s] for s in -N:N )
end

function energy(x::UniformIsing, s)
    E_edges = 0.0
    for i in 1:x.N
        for j in 1:i-1
            E_edges += s[i]*s[j]
        end
    end
    E_fields = dot(s, x.h)
    -x.J/x.N*E_edges - E_fields
end

# p(σᵢ = +1)
function marginals!(p, x::UniformIsing; Z = normalization(x))
    @unpack N, J, h, β, L, R = x
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
marginals(x::UniformIsing{T}; kw...) where T = marginals!(zeros(T,x.N), x; kw...)