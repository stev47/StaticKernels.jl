using Base: require_one_based_indexing

"""
    Kernel{X}(wf)

Create a stack-allocated kernel with axes `X`, wrapping a window function
`wf`.
The window function defines a reduction of values within the `X`-sized window.
For best performance you should annotate `wf` with `@inline` and index access
with `@inbounds`.

```@example
@inline function wf(w)
    return @inbounds w[0,-1] + w[-1,0] + 4*w[0,0] + w[1,0] + w[0,1]
end
Kernel{(-1:1,-1:1)}(wf)
```
"""
struct Kernel{X, F}
    wf::F
    function Kernel{X}(wf::F) where {X, F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes"))
        return new{X, F}(wf)
    end
end

"""
    Window{T,N,X} <: AbstractArray{T,N}

A stack-allocated view with axes `X` and cartesian indexing relative to some
position in the parent array.
"""
# FIXME: https://github.com/JuliaLang/julia/pull/32105
#struct Window{T,N,X} <: AbstractArray{T,N}
struct Window{T,N,X}
    position::CartesianIndex{N}
    parent_ptr::Ptr{T}
    parent_size::NTuple{N,Int}

    # TODO: relax to StridedArrays
    function Window{X}(a::DenseArray{T,N}, pos::CartesianIndex{N}) where {T,N,X}
        # because we use Base._sub2ind
        require_one_based_indexing(a)
        return new{T,N,X}(pos, pointer(a), size(a))
    end
end
