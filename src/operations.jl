function Base.map(k::Kernel, a::AbstractArray)
    b = boundary(k) isa BoundaryNone ?
        similar(a, eltype(a, k), size(a, k)) : similar(a, eltype(a, k))
    return map!(k, b, a)
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    outsize = boundary(k) isa BoundaryNone ? size(a, k) : size(a)
    size(b) == outsize ||
        throw(DimensionMismatch("$(size(b)) vs $(outsize)"))

    # these offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(a)))
    koffset = boundary(k) isa BoundaryNone ? CartesianIndex(first.(axes(k))) : zero(CartesianIndex{ndims(k)})

    @inline f(w) = @inbounds b[w.position + offset + koffset] = k(w)

    windowloop(f, k, a)

    return b
end
