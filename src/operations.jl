function Base.map(k::Kernel, a::AbstractArray)
    b = extension(k) isa ExtensionNone ?
        similar(a, eltype(k, a), size(k, a)) : similar(a, eltype(k, a))
    return map!(k, b, a)
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    outsize = extension(k) isa ExtensionNone ? size(k, a) : size(a)
    size(b) == outsize ||
        throw(DimensionMismatch("$(size(b)) vs $(outsize)"))

    # these offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(a)))
    koffset = extension(k) isa ExtensionNone ? CartesianIndex(first.(axes(k))) : zero(CartesianIndex{ndims(k)})

    @inline f(w) = @inbounds b[w.position + offset + koffset] = k(w)

    windowloop(f, k, a)

    return b
end
