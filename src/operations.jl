function Base.map(k::Kernel, a::AbstractArray; inner=false, kwargs...)
    b = inner ? similar(a, eltype(a, k), size(a, k)) : similar(a, eltype(a, k))
    return map!(k, b, a; inner=inner, kwargs...)
end

function Base.map!(k::Kernel, b::AbstractArray, a::AbstractArray; inner=false)
    outsize = inner ? size(a, k) : size(a)
    size(b) == outsize ||
        throw(DimensionMismatch("$(size(b)) vs $(outsize)"))

    # these offsets may constant propagate to 0
    offset = CartesianIndex(first.(axes(b))) - CartesianIndex(first.(axes(a)))
    koffset = inner ? CartesianIndex(first.(axes(k))) : zero(CartesianIndex{ndims(k)})

    @inline f(w) = @inbounds b[w.position + offset + koffset] = k(w)

    # awkward but gets the compiler to avoid allocation
    if inner
        windowloop(f, a, Val(axes(k)), Val{true}())
    else
        windowloop(f, a, Val(axes(k)), Val{false}())
    end

    return b
end
