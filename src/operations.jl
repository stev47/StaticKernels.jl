Base.map(k::Kernel, a::AbstractArray...) =
    map!(k, similar(first(a), eltype(k, a...), size(k, first(a))), a...)

@inline function _map_checkargs(k::Kernel, b::AbstractArray, a::AbstractArray...)
    allequal(axes.(Ref(k), a)) ||
        throw(DimensionMismatch("trimmed input axes don't agree: $(join(axes.(Ref(k), a), " vs "))"))

    size(b) == size(k, first(a)) ||
        throw(DimensionMismatch("output size doesn't match: $(size(b)) vs $(join(size.(Ref(k), a), " vs "))"))
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray...)
    _map_checkargs(k, b, a...)
    # these offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(k, first(a))))

    @inline f(w...) = @inbounds b[first(w).position + offset] = k(w...)
    windowloop(f, k, nothing, (_,_)->nothing, a...)

    return b
end

function Base.mapreduce(k::Kernel, op, a::AbstractArray...;
        dims=:, init=Base.reduce_empty(op, eltype(k, a...)))
    dims == Colon() ||
        throw(ArgumentError("dimension-wise reduction currently unsupported"))

    return mapfoldl(k, op, a...; init=init)
end

Base.mapfoldl(k::Kernel, op, a::AbstractArray...;
        init=Base.reduce_empty(op, eltype(k, a...))) = windowloop(k.f, k, init, op, a...)
