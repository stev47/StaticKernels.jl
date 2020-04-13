# Base function

Base.map(k::Kernel, a::AbstractArray) = map!(k, similar(a, eltype(a, k)), a)

function map_inner!(k::Kernel, b::AbstractArray, a::AbstractArray)
    axes(b) == axes(a) ||
        throw(DimensionMismatch("$(axes(b)) vs $(axes(a))"))

    @inbounds @simd ivdep for i in CartesianIndices(axes(a, k))
        w = Window{axes(k)}(a, i)
        b[i] = k(w)
    end

    return b
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    axes(b) == axes(a) ||
        throw(DimensionMismatch("$(axes(b)) vs $(axes(a))"))

    @inline f(w) = @inbounds b[w.position] = k(w)

    windowloop(f, a, Val(axes(k)))

    return b
end
