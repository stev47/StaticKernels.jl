using Test, BenchmarkTools
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

    @test x[axes(k, a)...] ≈ y
end

@testset "type stability" begin
    a = rand(100)
    @testset "scalar getindex" begin
        k = Kernel{(0:0,)}(w -> w[0], StaticKernels.ExtensionNone())
        @test eltype(k, a) == Float64
    end
    @testset "window" begin
        k = Kernel{(0:0,)}(w -> w, StaticKernels.ExtensionNone())
        @test eltype(k, a) <: Window
        @test isconcretetype(eltype(k, a))
    end
end

@testset "vector" begin
    a = rand(100)
    bnd = rand()
    k = Kernel{(0:1,)}(w -> something(w[1], bnd) - w[0], StaticKernels.ExtensionNothing())
    x = axes(k, a)
    b = map(k, a)
    c = diff(a)

    @test b[1:end-1] ≈ c
    @test b[end] ≈ bnd - a[end]
end

@testset "different return type" begin
    a = rand(10, 10)
    grad = Kernel{(0:1,0:1)}(w -> (w[1,0] - w[0,0], w[0,1] - w[0,0]))

    grada = map(grad, a)
    gx = axes(grad, a)
    @test diff(a, dims=1)[gx...] ≈ [x[1] for x in grada[gx...]]
    @test diff(a, dims=2)[gx...] ≈ [x[2] for x in grada[gx...]]
end

@testset "window as array" begin
    a = rand(3, 3)
    k = Kernel{(-1:1, -1:1)}(identity)
    w = Window{(-1:1, -1:1)}(k, a, CartesianIndex(2, 2))

    @test sum(Tuple(w)) ≈ sum(a)
end

@testset "memory allocations" begin
    a = rand(100)
    k = Kernel{(0:0,)}(w -> w[0], StaticKernels.ExtensionNone())
    b = similar(a, size(k, a))

    @test 0 == @ballocated map!($k, $b, $a)
    @test 0 == @ballocated sum($k, $a)
end
