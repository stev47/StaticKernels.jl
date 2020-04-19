Base.map(k::Kernel, a::AbstractArray) = map!(k, similar(a, eltype(k, a), size(k, a)), a)

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray)
    size(b) == size(k, a) ||
        throw(DimensionMismatch("$(size(b)) vs $(size(k, a))"))

    # this offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(k, a)))

    @inline f(w) = @inbounds b[w.position + offset] = k(w)
    windowloop(f, k, a, nothing, (_,_)->nothing)

    return b
end

function Base.mapreduce(k::Kernel, op, a::AbstractArray;
        dims=:, init=Base.reduce_empty(op, eltype(k, a)))
    dims == Colon() ||
        throw(ArgumentError("dimension-wise reduction currently unsupported"))

    @inline f(w) = @inbounds k(w)
    return windowloop(f, k, a, init, op)
end
