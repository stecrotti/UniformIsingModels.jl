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

x = UniformIsing(N, J, h, β)

@testset "normalization" begin
    _normaliz = (x, s) -> exp(-x.β*energy(x, s))
    Z_bruteforce = observables_bruteforce(x, [Obs(_normaliz)])[1]
    @test x.Z ≈ Z_bruteforce
end

@testset "magnetizations" begin
    m = site_magnetizations(x)
    _magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]) for i in 1:x.N]
    magnetiz_bruteforce = observables_bruteforce(x, _magnetiz)
    @test all(1:x.N) do i 
        m[i] ≈ magnetiz_bruteforce[i]
    end
end

@testset "pair magnetizations" begin
    p = pair_magnetizations(x)
    _pair_magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]*s[j]) 
                                        for i in 1:x.N for j in 1:x.N]
    pair_magnetiz_bruteforce = observables_bruteforce(x, _pair_magnetiz)
    @test all(Iterators.product(1:x.N,1:x.N)) do (i,j) 
        k = Int( (j-1)*N + i )
        p[i,j] ≈ pair_magnetiz_bruteforce[k]
    end
end

@testset "correlations" begin
    m = site_magnetizations(x)
    c = correlations(x)
    _correl = [Obs((x, s) -> pdf(x, s)*(s[i]*s[j]-m[i]*m[j])) 
                                        for i in 1:x.N for j in 1:x.N]
    correl_bruteforce = observables_bruteforce(x, _correl)
    @test all(Iterators.product(1:x.N,1:x.N)) do (i,j) 
        k = Int( (j-1)*N + i )
        isapprox( c[i,j], correl_bruteforce[k], atol=1e-4 )
    end
end

@testset "average energy" begin
    U = avg_energy(x)
    _energy = Obs((x,s) -> pdf(x,s)*energy(x,s))
    avg_energy_bruteforce = observables_bruteforce(x, [_energy])[1]
    @test U ≈ avg_energy_bruteforce
end

@testset "entropy" begin
    S = entropy(x)
    _entropy = Obs((x,s) -> -pdf(x,s)*log(pdf(x,s)))
    entropy_bruteforce = observables_bruteforce(x, [_entropy])[1]
    @test S ≈ entropy_bruteforce
end