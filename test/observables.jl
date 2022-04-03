# Loop once over all 2^N states and compute observables
function Obs(f::Function)
    o = 0.0
    function measure(x::UniformIsing, s) 
        o += f(x, s)
    end
end
function observables_bruteforce(x::UniformIsing, 
        observables::Vector{<:Function})
    if x.N > 10
        @warn "Exponential scaling alert"
    end
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

x = UniformIsing(N, J, h; β=β)

_normaliz = (x, s) -> exp(-x.β*energy(x, s))
obs_magnetiz = [Obs((x, s) -> exp(-x.β*energy(x, s))*s[i]) for i in 1:x.N]
obs_pair_magnetiz = [Obs((x, s) -> exp(-x.β*energy(x, s))*s[i]*s[j]) 
                                        for i in 1:x.N for j in 1:x.N]

obs_bruteforce = observables_bruteforce(x, vcat([Obs(_normaliz)], 
                                                 obs_magnetiz,
                                                 obs_pair_magnetiz))

Z_bruteforce = obs_bruteforce[1]
magnetiz_bruteforce = obs_bruteforce[1+1:1+x.N] ./ Z_bruteforce
pair_magnetiz_bruteforce = obs_bruteforce[1+x.N+1:end] ./ Z_bruteforce

@testset "normalization" begin
    Z = normalization(x)
    @test Z ≈ Z_bruteforce
end

@testset "magnetizations" begin
    m = site_magnetizations(x)
    @test all(1:x.N) do i 
        m[i] ≈ magnetiz_bruteforce[i]
    end
end

@testset "pair magnetizations" begin
    p = pair_magnetizations(x)
    @test all(Iterators.product(1:x.N,1:x.N)) do (i,j) 
        k = Int( (j-1)*N + i )
        p[i,j] ≈ pair_magnetiz_bruteforce[k]
    end
end