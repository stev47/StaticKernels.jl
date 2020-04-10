using Base: promote_op, @propagate_inbounds

"""
    Kernel{S}(wf)
    Kernel{S, C}(wf)

Create a stack-allocated kernel of size `S`, centered around `C` and wrapping a
window function `wf`.
The window function defines a reduction of values within an `S`-sized window
centered in `C`.
For best performance you should annotate `wf` with `@inline` and index access
with `@inbounds`.

```@example
@inline function wf(w)
    return @inbounds w[0,-1] + w[-1,0] + 4*w[0,0] + w[1,0] + w[0,1]
end
Kernel{(3,3)}(wf)
```
"""
struct Kernel{F, S, C}
    f::F
    function Kernel{S, C}(f::F) where {S, C, F}
        checkbounds(Bool, CartesianIndices(S), CartesianIndex(C)) ||
            throw(ArgumentError("kernel center $C out of bounds $S"))
        return new{F, S, C}(f)
    end
end

function Kernel{S}(f::F) where {S, F}
    all(isodd, S) ||
        throw(ArgumentError("please specify kernel center for non-centered kernels"))
    C = map(x -> div(x, 2, RoundUp), S)
    return Kernel{S, C}(f)
end

function Base.show(io::IO, ::MIME"text/plain", k::Kernel)
    println(io, "Kernel{$(size(k)),$(center(k))} with window function\n")
    print(code_lowered(k.f, (AbstractArray{Any},))[1])
end

Base.ndims(k::Kernel) = length(size(k))
Base.size(::Kernel{<:Any,S}) where S = S
# TODO: should return CartesianIndex
center(::Kernel{<:Any,<:Any,C}) where C = C
Base.eltype(a::AbstractArray{T,N}, k::Kernel) where {T,N} = Base.promote_op(k.f, Window{T,N,typeof(k)})

"""
    axes(a::AbstractArray, k::Kernel)

Returns axes along which `k` fits within `a`.
"""
function Base.axes(a::AbstractArray, k::Kernel)
    ndims(a) == ndims(k) ||
        throw(ArgumentError("mismatching number of dimensions: $(ndims(a)) vs $(ndims(k))"))
    return map(axes(a), size(k), center(k)) do ax, ks, kc
        first(ax) + kc - 1 : last(ax) - ks + kc
    end
end


"""
    getindex(a::AbstractArray, k::Kernel, i...)

Evaluate kernel `k` on `a` centered at index `i`.
"""
@inline @propagate_inbounds Base.getindex(a::DenseArray, k::Kernel, i...) =
    k(Window(a, k, CartesianIndex(i)))
