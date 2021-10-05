using Test, BenchmarkTools
using StaticKernels

using StaticKernels: Window

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1


@testset "consistency" begin
    a = rand(10, 10)
    a2 = rand(10, 10)

    @testset "window" begin
        b = rand(size(a)...)
        c = CartesianIndex(2, 2)
        k = Kernel{(-1:1, -1:1)}(identity)
        w = Window{axes(k)}(k, a, c)

        # AbstractArray interface
        @test ndims(w) == ndims(k)
        @test eltype(w) == eltype(a)

        # getindex / setindex
        for i in CartesianIndices(w)
            @inferred w[i]
            @test w[i] == a[c + i]
            w[i] = b[c + i]
            @test a[c + i] == w[i]
        end

        aw = a[1:3, 1:3]
        # iteration
        @test_broken sum(v for v in w) == sum(aw)

        # mapreduce
        @test sum(w) == sum(aw)
        @test minimum(w) == minimum(aw)

        # wrong kernel axes
        let a = rand(1)
            k = Kernel{(0:0,)}(w -> w[1])
            @test_throws MethodError map(k, a)
        end
    end

    @testset "extended array" begin
        ae = extend(a, StaticKernels.ExtensionNothing())
        #@test isnothing(ae[0,0])
    end

    @testset "kernel" begin
        k = Kernel{(-1:1,-1:1)}(@inline function(w) sum(Tuple(w)) end)
        k2 = Kernel{(-1:1,-1:1)}(@inline function(w1, w2) sum(Tuple(w1)) + prod(Tuple(w2)) end)
        @test eltype(k, a) == eltype(a)
        @test eltype(k2, a, a2) == eltype(a)
    end

    @testset "map" begin
        k = Kernel{(1:1, -1:-1)}(w -> w[1, -1])
        @test map(k, a) == a

        k = Kernel{(-1:1,0:1)}(w -> w[-1,0] + w[1,1])
        @test map(k, a) == a[1:end-2,1:end-1] .+ a[3:end,2:end]

        k = Kernel{(-1:1,0:1)}((w1, w2) -> w1[-1,0] + w2[1,1])
        @test map(k, a, a2) == a[1:end-2,1:end-1] .+ a2[3:end,2:end]
    end

    @testset "mapreduce" begin
        k = Kernel{(-1:1,0:1)}(w -> w[-1,0] + w[1,1])
        @test mapreduce(k, +, a) ≈ sum(a[1:end-2,1:end-1] .+ a[3:end,2:end])

        k = Kernel{(-1:1,0:1)}((w1, w2) -> w1[-1,0] + w2[1,1])
        @test mapreduce(k, +, a, a2) ≈ sum(a[1:end-2,1:end-1] .+ a2[3:end,2:end])

        k = Kernel{(0:1, 0:0)}(w -> sum(Tuple(w)))
        @test sum(k, a) ≈ sum(a[1:end-1,:]) + sum(a[2:end,:])
    end

    @testset "kernel macro" begin
        b = vec(a)

        k1 = @kernel w -> w[0]
        k2 = Kernel{(0:0,)}(w -> w[0])
        @test map(k1, b) == map(k2, b)

        k1 = @kernel w -> w[1] + w[-2]
        k2 = Kernel{(-2:1,)}(w -> w[1] + w[-2])
        @test map(k1, b) == map(k2, b)

        k1 = @kernel w -> w[1,2] + w[-3,5]
        k2 = Kernel{(-3:1,2:5)}(w -> w[1,2] + w[-3,5])
        @test map(k1, a) == map(k2, a)

        @test_throws Exception @macroexpand @kernel w -> w
        @test_throws Exception @macroexpand @kernel w -> w[1] + w[2,3]
    end

    @testset "gradient" begin
        grad = Kernel{(0:1,0:1)}(w -> (w[1,0] - w[0,0], w[0,1] - w[0,0]))

        grada = map(grad, a)
        gx = axes(grad, a)
        @test diff(a, dims=1)[gx...] ≈ [x[1] for x in grada[gx...]]
        @test diff(a, dims=2)[gx...] ≈ [x[2] for x in grada[gx...]]
    end

    @testset "extensions" begin
        ae = extend(a, StaticKernels.ExtensionNone())
        k = Kernel{(-1:1, -1:1)}(w -> sum(Tuple(w)))
        @test size(map(k, ae)) == size(k, ae)

        ae = extend(a, StaticKernels.ExtensionNothing())
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0] + w[0,0])
        @test_throws MethodError map(k, ae)
        k = Kernel{(-1:1, -1:1)}(w -> something(w[1,0], 0))
        @test size(map(k, ae)) == size(k, ae)
        @test all(map(k, ae)[end, :] .== 0)

        ae = extend(a, StaticKernels.ExtensionReplicate())
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0])
        @test size(map(k, ae)) == size(k, ae)
        @test all(map(k, ae)[end, :] .== ae[end, :])

        ae = extend(a, StaticKernels.ExtensionCircular())
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0])
        @test size(map(k, ae)) == size(k, ae)
        @test all(map(k, ae)[end, :] .== ae[begin, :])

        ae = extend(a, StaticKernels.ExtensionSymmetric())
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0])
        @test size(map(k, ae)) == size(k, ae)
        @test all(map(k, ae)[end, :] .== ae[end-1, :])

        ae = extend(a, StaticKernels.ExtensionConstant(0.))
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0])
        @test eltype(k, ae) == eltype(ae)
        @test size(map(k, ae)) == size(k, ae)
        @test all(map(k, ae)[end, :] .== 0)

        ae = extend(a, StaticKernels.ExtensionConstant(missing))
        k = Kernel{(-1:1, -1:1)}(w -> w[1,0])
        @test eltype(k, ae) == Union{Missing,eltype(ae)}
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
        @test eltype(k, a) == eltype(a)

        k = Kernel{(0:0,)}((w1, w2) -> w1[0] + w2[0])
        @test eltype(k, a, a) == eltype(a)
        @test eltype(k, a, round.(Int, a)) == Base.promote_type(eltype(a), Int)

        k = Kernel{(0:0,)}(w -> Tuple(w))
        @test eltype(k, a) <: Tuple
        @test isconcretetype(eltype(k, a))
    end
end

@testset "memory allocations" begin
    a = rand(100)
    a2 = rand(100)
    ks = [
        (Kernel{(0:0,)}(w -> w[0]), StaticKernels.ExtensionNone()),
        (Kernel{(0:1,)}(w -> something(w[1], 0.)), StaticKernels.ExtensionNothing()),
        (Kernel{(0:1,)}(w -> w[1]), StaticKernels.ExtensionReplicate()),
        (Kernel{(0:1,)}(w -> w[1]), StaticKernels.ExtensionCircular()),
        (Kernel{(0:1,)}(w -> w[1]), StaticKernels.ExtensionSymmetric()),
        (Kernel{(0:1,)}(w -> w[1]), StaticKernels.ExtensionConstant(0.)),
        (Kernel{(0:1,)}(w -> Tuple(w)), StaticKernels.ExtensionNone())]

    @testset "map! $(x[2])" for x in ks
        k, extension = x
        ae = extend(a, extension)
        b = similar(ae, eltype(k, ae), size(k, ae))
        @test 0 == @ballocated map!($k, $b, $ae)
    end

    @testset "sum $(x[2])" for x in ks[1:6]
        k, extension = x
        ae = extend(a, extension)
        @test eltype(k, ae) == eltype(a)
        @test 0 == @ballocated sum($k, $ae)
    end

    @testset "multiargument" begin
        k = Kernel{(-1:1,)}((w1, w2) -> w1[-1] + w2[1])
        b = similar(a, eltype(k, a, a2), size(k, a))
        @test 0 == @ballocated map!($k, $b, $a, $a2)
        @test 0 == @ballocated mapreduce($k, +, $a, $a2)
    end
end

@testset "performance" begin
    a = rand(1000000)
    a2 = rand(1000, 1000)
    a2ext = extend(rand(1000, 1000), StaticKernels.ExtensionConstant(0))

    @testset "Base.diff" begin
        k = Kernel{(0:1,)}(@inline function(w) @inbounds w[1] - w[0] end)

        @test 1.3 > @belapsed(map($k, $a)) / @belapsed(diff($a))
    end

    @testset "Base.map" begin
        k = Kernel{(0:0,)}(@inline function(w) @inbounds w[0] end)
        b = similar(a, size(k, a))

        @test 1.1 > @belapsed(map!($k, $b, $a)) / @belapsed(map!(identity, $b, $a))
    end

    @testset "window mapreduce" begin
        k1 = Kernel{(-1:1,)}(@inline function(w) @inbounds sum(w) end)
        k2 = Kernel{(-1:1,)}(@inline function(w) @inbounds w[-1] + w[0] + w[1] end)
        b = similar(a, size(k1, a))

        @test 1.1 > @belapsed(map!($k2, $b, $a)) / @belapsed(map!($k1, $b, $a))
    end

    @testset "extension" begin
        k = Kernel{(-1:1,-1:1)}(
            @inline function(w) @inbounds w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1] end)
        b2 = similar(a2, size(k, a2))
        b2ext = similar(a2ext, size(k, a2ext))

        @test 1.1 > @belapsed(map!($k, $b2ext, $a2ext)) / @belapsed(map!($k, $b2, $a2))
    end
end

