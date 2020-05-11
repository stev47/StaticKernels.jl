using Base: promote_op, @propagate_inbounds

function Base.show(io::IO, ::MIME"text/plain", k::Kernel)
    print(io, "Kernel{$(axes(k))} with window function $(k.f)")
    #print(code_lowered(k.f, (AbstractArray{Any},))[1])
end

Base.axes(::Kernel{X}) where X = X
Base.ndims(k::Kernel) = length(axes(k))
Base.size(k::Kernel) = length.(axes(k))
Base.keys(k::Kernel) = CartesianIndices(axes(k))

@inline (k::Kernel)(w::Window...) = k.f(w...)

# FIXME: revise the following questionable interface

"""
    eltype(k::Kernel, a::AbstractArray)

Infer the return type of `k` when applied to an interior window of `a`.
"""
Base.eltype(k::Kernel, a::AbstractArray...) = promote_op(k.f, map(ai -> wintype(k, ai), a)...)

wintype(k::Kernel, a::AbstractArray) =
    Window{eltype(a),ndims(a),axes(k),typeof(k),typeof(a)}
wintype(k::Kernel, a::ExtensionArray) =
    Window{eltype(a),ndims(a),axes(k),typeof(k),
    Array{Base.promote_type(eltype(a),eltype_extension(a)),ndims(a)}}

"""
    axes(k::Kernel, a::AbstractArray)

Return axes along which `k` can be applied to a window of `a`.
"""
@inline function Base.axes(k::Kernel, a::AbstractArray)
    ndims(a) == ndims(k) ||
        throw(DimensionMismatch("$(ndims(a)) vs $(ndims(k))"))

    extension(a) != ExtensionNone() && return axes(a)

    return map(axes(a), axes(k)) do ax, kx
        first(ax) - first(kx) : last(ax) - last(kx)
    end
end

"""
    size(k::Kernel, a::AbstractArray)

Return size of the cartesian region over which `k` can be applied to a window
of `a`.
"""
@inline Base.size(k::Kernel, a::AbstractArray) = length.(axes(k, a))
