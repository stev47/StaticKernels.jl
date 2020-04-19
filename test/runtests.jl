using Test, BenchmarkTools
using StaticKernels

using StaticKernels: Window

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1

a = rand(rand(5:10, 3)...)

dones = ntuple(i -> 1, ndims(a))
dzeros = ntuple(i -> 0, ndims(a))

@testset "consistency" begin
    @testset for i in 1:ndims(a)
        ddir = ntuple(j -> Int(j == i), ndims(a))

        k = Kernel{UnitRange.(dzeros, ddir)}() do w
            return something(w[ddir...], 0) - w[dzeros...]
        end

        @test isbits(k)

        x = map(k, a)
        y = diff(a, dims=i)

        @test x[axes(k, a)...] ≈ y
    end

    @testset "gradient" begin
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
end

@testset "type stability" begin
    a = rand(100)

    @testset "window" begin
        k = Kernel{(0:0,)}(w -> w)

        @test eltype(k, a) <: Window
        @test isconcretetype(eltype(k, a))
    end

    @testset "window scalar getindex" begin
        k = Kernel{(0:0,)}(w -> w[0])

        @test eltype(k, a) == Float64
    end

    @testset "window tuple" begin
        k = Kernel{(0:0,)}(w -> Tuple(w))

        @test eltype(k, a) <: Tuple
        @test isconcretetype(eltype(k, a))
    end
end

@testset "memory allocations" begin
    a = rand(100)
    k = Kernel{(0:0,)}(w -> w[0], StaticKernels.ExtensionNone())

    @testset "map!" begin
        b = similar(a, size(k, a))
        @test 0 == @ballocated map!($k, $b, $a)
    end

    @testset "sum!" begin
        @test 0 == @ballocated sum($k, $a)
    end
end

@testset "performance" begin
    @testset "Base.diff" begin
        a = rand(10000)
        k = Kernel{(0:1,)}((@inline function(w) @inbounds w[1] - w[0] end))

        @test 1.3 > @belapsed(map($k, $a)) / @belapsed(diff($a))
    end
end

