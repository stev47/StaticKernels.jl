using Test
using StaticFilters

using StaticFilters: Window


a = rand(rand(5:10, 3)...)

dones = ntuple(i -> 1, ndims(a))
dzeros = ntuple(i -> 0, ndims(a))

@testset "basics" for i in 1:ndims(a)
    ddir = ntuple(j -> Int(j == i), ndims(a))

    ksize = dones .+ Tuple(ddir)
    k = Kernel{ksize, dones}() do w
        return w[ddir...] - w[dzeros...]
    end

    @test isbits(k)

    x = map(k, a)
    y = diff(a, dims=i)

    @test x[axes(a,k)...] == y
end

@testset "different return type" begin
    a = rand(10, 10)
    grad = Kernel{(2,2),(1,1)}(w -> (w[1,0] - w[0,0], w[0,1] - w[0,0]))

    grada = map(grad, a)
    gx = axes(a, grad)
    @test diff(a, dims=1)[gx...] == [x[1] for x in grada[gx...]]
    @test diff(a, dims=2)[gx...] == [x[2] for x in grada[gx...]]
end

@testset "window as array" begin
    a = rand(3, 3)
    k = Kernel{(3, 3)}(w -> sum(w))
    w = Window(a, k, CartesianIndex(2, 2))

    @test sum(w) == sum(a)
end
