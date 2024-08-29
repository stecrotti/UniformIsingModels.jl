N = 10
J = 0.5
β = 2.3

x = UniformIsing(N, J, β)

@testset "basics" begin

    @testset "outer constructor" begin
        @test all(isequal(0), x.h)
    end

    @testset "left accumulator" begin
        L = UniformIsingModels.accumulate_left(x.h, x.β)
        @test L == x.L
    end

    @testset "mutate and recompute partials" begin
        hnew = ones(N)
        Jnew = -1.1
        βnew = 0.1
        x.h = hnew
        x.J = Jnew
        x.β = βnew
        recompute_partials!(x)
        xnew = UniformIsing(N, Jnew, hnew, βnew)
        @test lognormalization(x) == lognormalization(xnew)
    end 

end