using Base: promote_op, @propagate_inbounds

function Base.show(io::IO, ::MIME"text/plain", k::Kernel)
    println(io, "Kernel{$(axes(k))} with window function\n")
    print(code_lowered(k.wf, (AbstractArray{Any},))[1])
end

Base.axes(::Kernel{X}) where X = X
Base.ndims(k::Kernel) = length(axes(k))
Base.size(k::Kernel) = length.(axes(k))

"""
    eltype(a::AbstractArray, k::Kernel)

Infer the return type of `k` applied to a window of `a`.
"""
Base.eltype(a::AbstractArray{T,N}, k::Kernel) where {T,N} =
    Base.promote_op(k.wf, Window{T,N,axes(k)})

"""
    axes(a::AbstractArray, k::Kernel)

Returns axes along which `k` fits within `a`.
"""
@inline function Base.axes(a::AbstractArray, k::Kernel)
    ndims(a) == ndims(k) ||
        throw(DimensionMismatch("$(ndims(a)) vs $(ndims(k))"))

    return map(axes(a), axes(k)) do ax, kx
        first(ax) - first(kx) : last(ax) - last(kx)
    end
end

"""
    getindex(a::AbstractArray, k::Kernel, i...)

Evaluate kernel `k` on `a` centered at index `i`.
"""
@inline @propagate_inbounds Base.getindex(a::DenseArray, k::Kernel, i...) =
    k(Window(k, a, CartesianIndex(i)))

"""
    (k::Kernel)(w::Window)

Evaluates `k` on `w`.
If `k` does not fit within `w` and `k`'s window function leaks missing boundary
data then an error will be thrown.
"""
@inline function (k::Kernel)(w::Window)
    ret = k.wf(w)

    isa(ret, nonmissingtype(Base.promote_op(k.wf, typeof(w)))) ||
        throw(MissingException("evaluation of $(axes(k))-kernel on window $(axes(w)) leaks missing boundary data"))

    return ret
end
