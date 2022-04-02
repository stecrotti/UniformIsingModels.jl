N = 10
J = 2.0
h = randn(N)
β = 0.1

x = UniformIsing(N, J, h; β)

function Obs(f::Function)
    o = 0.0
    function measure(x::UniformIsing, s) 
        o += f(x, s)
    end
end

_normaliz = (x, s) -> exp(-x.β*energy(x, s))
_energy = (x, s) -> energy(x, s)
_obs_marginals = [Obs((x, s) -> exp(-x.β*energy(x, s))*(s[i]==1)) for i in 1:x.N]

function observables_bruteforce(x::UniformIsing, 
        observables::Vector{<:Function})
    for s in Iterators.product(fill((-1,1),x.N)...)
        for f! in observables  
            f!(x, s)
        end
    end
    [obs.o.contents for obs in observables]
end

observables = vcat([Obs(_normaliz), Obs(_energy)], obs_marginals)
obs_bruteforce = observables_bruteforce(x, observables)

Z_bruteforce = obs_bruteforce[1]
E_bruteforce = obs_bruteforce[2]
marginals_bruteforce = obs_bruteforce[2+1:2+x.N] ./ Z_bruteforce

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