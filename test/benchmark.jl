using Test, BenchmarkTools

using StaticArrays: SVector
using StaticKernels: Kernel
using NNlib: DenseConvDims, conv!
using ImageFiltering: centered, imfilter!, Inner
using LocalFilters: convolve!
import LocalFilters
using OffsetArrays: OffsetArray
using LoopVectorization: @avx

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
pd = 19

kernel_axes(k) = map(n -> -n÷2:n÷2, size(k))
inner_axes(k, a) = map(axes(a), kernel_axes(k)) do ax, kx
        first(ax) - first(kx) : last(ax) - last(kx)
    end
inner_size(k, a) = length.(inner_axes(k, a))

function bench_linfilt(k, a, ::Val{:StaticKernels})
    print(rpad("  StaticKernels", pd))

    ktuple = Tuple(k)
    @inline function wf(w)
        return sum(SVector(Tuple(w)) .* SVector(ktuple))
    end
    @assert size(k, 1) == size(k, 2)
    K = size(k, 1)
    kern = Kernel{kernel_axes(k)}(wf)
    b1 = similar(a, size(kern, a))
    @btime map!($kern, $b1, $a)
    b1
end

function bench_linfilt(k, a, ::Val{:LoopVectorization})
    print(rpad("  LoopVectorization", pd))

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

    b5 = similar(a, inner_axes(k, a))
    k5 = reshape(k, kernel_axes(k))
    @btime $filter2davx!($b5, $a, $k5)
    #filter2davx!(b5, a, k5)
    # weird OffsetArray broadcast problem
    b5 = parent(b5)
end

function bench_linfilt(k, a, ::Val{:NNlib})
    print(rpad("  NNlib", pd))

    ar = reshape(a, (size(a)..., 1, 1))
    b2 = similar(a, (inner_size(k, a)..., 1, 1))
    kr = reshape(k, (size(k)..., 1, 1))
    cdims = DenseConvDims(ar, kr; stride=1, padding=0, dilation=1, flipkernel=true)
    @btime conv!($b2, $ar, $kr, $cdims)
    b2
end

function bench_linfilt(k, a, ::Val{:ImageFiltering})
    print(rpad("  ImageFiltering", pd))

    b3 = similar(a, inner_axes(k, a))
    kc = centered(k)
    @btime imfilter!($b3, $a, $kc, Inner())
    # weird OffsetArray broadcast problem
    b3 = parent(b3)
end

function bench_linfilt(k, a, ::Val{:LocalFilters})
    print(rpad("  LocalFilters", pd))

    b4 = similar(a)
    # mathematical convolution
    kr = LocalFilters.Kernel(reverse(reverse(k, dims=1), dims=2))
    @btime convolve!($b4, $a, $kr)
    # no way to only compute inner part
    b4 = b4[inner_axes(k, a)...]
end


# Linear Filtering

for N in (100, 1000), K in (3, 5, 7)
    a = rand(N, N)
    k = rand(K, K)

    println("Array: $(size(a)), Kernel: $(size(k))")

    b = bench_linfilt(k, a, Val(:StaticKernels))

    # disabled: :LocalFilters
    for s in [:LoopVectorization, :NNlib, :ImageFiltering]
        @test b ≈ bench_linfilt(k, a, Val(s))
    end
end
