N = 20
J = 2.5
h = 0.3*randn(N)
β = 0.1

x = UniformIsing(N, J, h, β)

nsamples = 10^3
ntests = 20
@testset "sampling" begin
    X = [sample(x)[1] for _ in 1:nsamples]
    σ_test = [rand((-1,1), N) for _ in 1:ntests]
    tol = 1 / sqrt(nsamples)
    @test all(1:ntests) do t
        σ = σ_test[t]
        p = pdf(x, σ)
        p_empirical = sum(x == σ for x in X) / nsamples
        abs(p - p_empirical) < tol
    end
end