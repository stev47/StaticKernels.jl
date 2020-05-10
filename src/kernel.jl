using Base: promote_op, @propagate_inbounds

"""
    Kernel{X}(f)

Create a kernel with axes `X` wrapping a kernel function `f`.

The kernel function `f` defines a reduction of values within an `X`-sized
view. When the kernel is applied to data of an array `a` the kernel function
gets called with one argument `w::Window` that provides a local view on the
data.

```@example
# Laplacian 3x3 Kernel (i.e. axes (-1:1, -1:1))
function f(w)
    return w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1]
end
Kernel{(-1:1,-1:1)}(f)
```

For best performance you should annotate the kernel function `f` with `@inline`
and index accesses within using `@inbounds`.
"""
function Kernel end

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
Base.eltype(k::Kernel, a::AbstractArray...) =
    promote_op(k.f, map(ai->Window{eltype(ai),ndims(ai),axes(k),typeof(k),typeof(ai)}, a)...)

Base.eltype(k::Kernel, a::ExtensionArray{T,N}) where {T,N} =
    promote_op(k.f, Window{T,N,axes(k),typeof(k),Array{Base.promote_type(T,eltype_extension(a)),ndims(a)}})

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
