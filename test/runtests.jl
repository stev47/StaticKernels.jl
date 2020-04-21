using Test, BenchmarkTools
using StaticKernels

using StaticKernels: Window

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1


@testset "consistency" begin
    a = rand(10, 10)

    @testset "gradient" begin
        grad = Kernel{(0:1,0:1)}(w -> (w[1,0] - w[0,0], w[0,1] - w[0,0]))

        grada = map(grad, a)
        gx = axes(grad, a)
        @test diff(a, dims=1)[gx...] ≈ [x[1] for x in grada[gx...]]
        @test diff(a, dims=2)[gx...] ≈ [x[2] for x in grada[gx...]]
    end

    @testset "window" begin
        b = similar(a)
        c = CartesianIndex(2, 2)
        k = Kernel{(-1:1, -1:1)}(identity)
        w = Window{axes(k)}(k, a, c)

        # getindex / setindex
        for i in CartesianIndices(axes(k))
            @inferred w[i]
            @test w[i] == a[c + i]
            w[i] = b[c + i]
            @test a[c + i] == w[i]
        end
    end
end

@testset "type stability" begin
    a = rand(100)

    @testset "window" begin
        k = Kernel{(0:0,)}(w -> w)

        @test eltype(k, a) <: Window
        @test isconcretetype(eltype(k, a))
    end

    @testset "kernel evaluation" begin
        k = Kernel{(0:0,)}(w -> w[0])
        @test eltype(k, a) == Float64

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
    a = rand(1000000)

    @testset "Base.diff" begin
        k = Kernel{(0:1,)}((@inline function(w) @inbounds w[1] - w[0] end))

        @test 1.3 > @belapsed(map($k, $a)) / @belapsed(diff($a))
    end

    @testset "Base.map" begin
        k = Kernel{(0:0,)}((@inline function(w) @inbounds w[0] end))
        b = similar(a, size(k, a))

        @test 1.1 > @belapsed(map!($k, $b, $a)) / @belapsed(map!(identity, $b, $a))
    end
end

