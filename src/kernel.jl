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
    println(io, "Kernel{$(axes(k))} with window function\n")
    print(code_lowered(k.f, (AbstractArray{Any},))[1])
end

Base.axes(::Kernel{X}) where X = X
Base.ndims(k::Kernel) = length(axes(k))
Base.size(k::Kernel) = length.(axes(k))

@inline (k::Kernel)(w::Window) = k.f(w)

# FIXME: revise the following questionable interface

"""
    eltype(a::AbstractArray, k::Kernel)

Infer the return type of `k` applied to a window of `a`.
"""
Base.eltype(a::AbstractArray{T,N}, k::Kernel) where {T,N} =
    promote_op(k.f, Window{T,N,axes(k)})

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
    size(a::AbstractArray, k::Kernel)

Returns size of `a` in which `k` can fit.
"""
@inline Base.size(a::AbstractArray, k::Kernel) = length.(axes(a, k))
