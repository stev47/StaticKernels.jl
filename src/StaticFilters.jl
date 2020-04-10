module StaticFilters

export Kernel

include("kernel.jl")
include("window.jl")


Base.map(k::Kernel, a::AbstractArray) = map!(k, similar(a), a)

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    axes(b) == axes(a) ||
        throw(ArgumentError("non-matching dimensions $(axes(b)) vs $(axes(a))"))

    @inbounds @simd for i in CartesianIndices(axes(a, k))
        b[i] = a[k, i]
    end

    return b
end


end # module
