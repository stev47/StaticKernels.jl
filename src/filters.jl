# Base function

Base.map(k::Kernel, a::AbstractArray) = map!(k, similar(a, eltype(a, k)), a)

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    axes(b) == axes(a) ||
        throw(ArgumentError("non-matching dimensions $(axes(b)) vs $(axes(a))"))

    @inbounds @simd for i in CartesianIndices(axes(a, k))
        b[i] = a[k, i]
    end

    return b
end
