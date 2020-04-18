function Base.map(k::Kernel, a::AbstractArray)
    b = boundary(k) isa BoundaryNone ?
        similar(a, eltype(k, a), size(k, a)) : similar(a, eltype(k, a))
    return map!(k, b, a)
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    outsize = boundary(k) isa BoundaryNone ? size(k, a) : size(a)
    size(b) == outsize ||
        throw(DimensionMismatch("$(size(b)) vs $(outsize)"))

    # these offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(a)))
    koffset = boundary(k) isa BoundaryNone ? CartesianIndex(first.(axes(k))) : zero(CartesianIndex{ndims(k)})

    @inline f(w) = @inbounds b[w.position + offset + koffset] = k(w)

    windowloop(f, k, a)

    return b
end
