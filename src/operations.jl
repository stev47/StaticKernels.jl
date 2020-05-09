Base.map(k::Kernel, a::AbstractArray...) =
    map!(k, similar(first(a), eltype(k, a...), size(k, first(a))), a...)

allequal(a) = isempty(a) ? true : mapfoldl(==(first(a)), &, a)

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray...)
    allequal(axes.(Ref(k), a)) ||
        throw(DimensionMismatch("input axes don't agree $(join(axes.(Ref(k), a), " vs "))"))

    size(b) == size(k, first(a)) ||
        throw(DimensionMismatch("$(size(b)) vs $(join(size.(Ref(k), a), " vs "))"))

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
        init=Base.reduce_empty(op, eltype(k, a...))) = windowloop(k, k, init, op, a...)
