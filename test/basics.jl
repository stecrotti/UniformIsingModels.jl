N = 10
J = 0.5
β = 2.3

x = UniformIsing(N, J, β)

@testset "basics" begin

    @testset "outer constructor" begin
        @test all(isequal(0), x.h)
    end

    @testset "recompute partials" begin
        hnew = ones(N)
        x.h = hnew
        recompute_partials!(x)
        xnew = UniformIsing(N, J, hnew, β)
        @test lognormalization(x) == lognormalization(xnew)
    end 

end