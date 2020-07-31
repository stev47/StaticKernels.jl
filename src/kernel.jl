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

"""
    @kernel(wf)

Create a `Kernel` from the supplied anonymous window function `wf` by inferring
the kernel axes automatically.

For example the following two lines create functionally equivalent kernels:
```julia
@kernel w -> w[1] - w[0]
Kernel{(0:1,)}(w -> w[1] - w[0])
```

This macro currently expects an anonymous function and is intended purely as a
shorthand in the frequent case that your kernel axes can easily be inferred
from constant indices.
"""
macro kernel(wf)
    # TODO: automatically add @inline and @inbounds
    function walk(f, expr)
        f(expr)
        isa(expr, Expr) || return
        foreach(x -> walk(f, x), expr.args)
    end
    enclose(a) = first(a) : last(a)
    enclose(a, b) = min(first(a), first(b)) : max(last(a), last(b))

    wf.head == :-> || error("anonymous function expected")

    if wf.args[1] isa Symbol
        wfargs = [wf.args[1]]
    elseif wf.args[1] isa Expr && wf.args[1].head == :tuple
        wfargs = wf.args[1].args
    else
        error("unexpected function arguments")
    end
    wfbody = wf.args[2]

    d = nothing
    ax = nothing
    walk(wfbody) do x
        if x isa Expr && x.head == :ref && x.args[1] in wfargs
            all(x -> isa(x, Int), x.args[2:end]) ||
                error("encountered non-explicit index, consider writing explicit indices or using the non-macro syntax instead")
            if isnothing(d)
                d = length(x.args) - 1
                ax = enclose.(x.args[2:end])
            else
                d == length(x.args) - 1 || error("window index dimensions don't match")
                ax = enclose.(ax, x.args[2:end])
            end
        end
    end
    (isnothing(d) || isnothing(ax)) &&
        error("could not determine kernel axes")
    return :( $Kernel{($(ax...),)}($(esc(wf))) )
end
