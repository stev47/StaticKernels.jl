using Test
using StaticKernels

using StaticKernels: Window


a = rand(rand(5:10, 3)...)

dones = ntuple(i -> 1, ndims(a))
dzeros = ntuple(i -> 0, ndims(a))

@testset "basics" for i in 1:ndims(a)
    ddir = ntuple(j -> Int(j == i), ndims(a))

    k = Kernel{UnitRange.(dzeros, ddir)}() do w
        return something(w[ddir...], 0) - w[dzeros...]
    end

    @test isbits(k)

    x = map(k, a)
    y = diff(a, dims=i)

    @test x[axes(a,k)...] == y
end

@testset "vector" begin
    a = rand(100)
    k = Kernel{(0:1,)}(w -> something(w[1], 0) - w[0])
    x = axes(a, k)
    b = map(k, a)
    c = diff(a)

    @test b[x...] == c[x...]
end

@testset "different return type" begin
    a = rand(10, 10)
    grad = Kernel{(0:1,0:1)}(w -> (something(w[1,0], 0) - w[0,0], something(w[0,1], 0) - w[0,0]))

    grada = @inferred map(grad, a)
    gx = axes(a, grad)
    @test diff(a, dims=1)[gx...] == [x[1] for x in grada[gx...]]
    @test diff(a, dims=2)[gx...] == [x[2] for x in grada[gx...]]
end

@testset "window as array" begin
    a = rand(3, 3)
    w = Window{(-1:1, -1:1)}(a, CartesianIndex(2, 2))

    @test sum(Tuple(w)) == sum(a)
end
