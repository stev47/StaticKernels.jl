using Test, BenchmarkTools

using StaticArrays: SVector
using StaticKernels: Kernel
using NNlib: DenseConvDims, conv!
using ImageFiltering: centered, imfilter!, Inner
using LocalFilters: convolve!
import LocalFilters
using OffsetArrays: OffsetArray
using LoopVectorization: @avx

# from discourse forum:
# https://discourse.julialang.org/t/ann-statickernels-jl-fast-kernel-operations-on-arrays/37658
function filter2davx!(out::AbstractMatrix, A::AbstractMatrix, kern)
    @avx for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J] * kern[I]
        end
        out[J] = tmp
    end
    out
end

# Linear Filtering

pd = 19

for N in (10, 100, 1000), K in (3, 5, 7)
    a = rand(N, N)
    k = rand(K, K)

    println("Array: $(size(a)), Kernel: $(size(k))")


    # StaticKernels.jl
    print(rpad("  StaticKernels", pd))

    ktuple = Tuple(k)
    @inline function wf(w)
        return sum(SVector(Tuple(w) .* ktuple))
    end
    kern = Kernel{(-K÷2:K÷2, -K÷2:K÷2)}(wf)
    b1 = similar(a, size(a, kern))
    @btime map!($kern, $b1, $a)


    # LoopVectorization.jl
    print(rpad("  LoopVectorization", pd))

    b5 = similar(a, axes(a, kern))
    k5 = reshape(k, axes(kern))
    @btime filter2davx!($b5, $a, $k5)
    # weird OffsetArray broadcast problem
    b5 = parent(b5)


    # NNlib.jl
    print(rpad("  NNlib", pd))

    ar = reshape(a, (size(a)..., 1, 1))
    b2 = similar(a, (size(a, kern)..., 1, 1))
    kr = reshape(k, (size(k)..., 1, 1))
    cdims = DenseConvDims(ar, kr; stride=1, padding=0, dilation=1, flipkernel=true)
    @btime conv!($b2, $ar, $kr, $cdims)


    # ImageFiltering.jl
    print(rpad("  ImageFiltering", pd))

    b3 = similar(a, axes(a, kern))
    kc = centered(k)
    @btime imfilter!($b3, $a, $kc, Inner())
    # weird OffsetArray broadcast problem
    b3 = parent(b3)


    # LocalFilters.jl
    print(rpad("  LocalFilters", pd))

    b4 = similar(a)
    # mathematical convolution
    kr = LocalFilters.Kernel(reverse(reverse(k, dims=1), dims=2))
    @btime convolve!($b4, $a, $kr)
    # no way to only compute inner part
    b4 = b4[axes(a, kern)...]


    @test b1 ≈ b2 ≈ b3 ≈ b4 ≈ b5
end
