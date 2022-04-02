# Loop once over all 2^N states and compute observables
function Obs(f::Function)
    o = 0.0
    function measure(x::UniformIsing, s) 
        o += f(x, s)
    end
end
function observables_bruteforce(x::UniformIsing, 
        observables::Vector{<:Function})
    for s in Iterators.product(fill((-1,1),x.N)...)
        for f! in observables  
            f!(x, s)
        end
    end
    [obs.o.contents for obs in observables]
end

N = 10
J = 0.5
h = 1.2*randn(N)
β = 2.3

x = UniformIsing(N, J, h; β)

_normaliz = (x, s) -> exp(-x.β*energy(x, s))
obs_marginals = [Obs((x, s) -> exp(-x.β*energy(x, s))*(s[i]==1)) for i in 1:x.N]

obs_bruteforce = observables_bruteforce(x, vcat([Obs(_normaliz)], obs_marginals))

Z_bruteforce = obs_bruteforce[1]
marginals_bruteforce = obs_bruteforce[1+1:1+x.N] ./ Z_bruteforce

@testset "normalization" begin
    Z = normalization(x)
    @test Z ≈ Z_bruteforce
end

@testset "marginals" begin
    p = marginals(x)
    @test all(1:x.N) do i 
        p[i] ≈ marginals_bruteforce[i]
    end
end