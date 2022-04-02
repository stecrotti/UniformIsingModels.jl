N = 10
J = 2.5
h = 0.3*randn(N)
β = 0.1

x = UniformIsing(N, J, h; β=β)

nsamples = 100
@testset "sampling" begin
    @test all(1:nsamples) do n
        σ, p = sample(x)
        p ≈ pdf(x, σ)
    end
end