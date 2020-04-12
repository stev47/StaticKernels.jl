using Base: require_one_based_indexing, _sub2ind, substrides

"""
    Window{T,N,X} <: AbstractArray{T,N}

A stack-allocated view with axes `X` and cartesian indexing relative to some
position in the parent array.
"""
struct Window{T,N,X} <: AbstractArray{T,N}
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

"""
    Window{X}(a::DenseArray, pos::CartesianIndex)
    Window(k::Kernel, a::DenseArray, pos::CartesianIndex)

Create a stack-allocated view on `a` with axes `X` and cartesian indexing
relative to `pos` in the parent array.
If constructed with a kernel `k` then `X = axes(k)`.

The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
function Window(k::Kernel, a::DenseArray, pos::CartesianIndex)
    ndims(k) == ndims(a) == length(pos) ||
        throw(DimensionMismatch("$(ndims(k)) vs $(ndims(a)) vs $(length(pos))"))

    return Window{axes(k)}(a, pos)
end

@inline function Base.Tuple(w::Window)
    ci = eachindex(w)
    @inline function f(i) @inbounds w[ci[i]] end
    return ntuple(f, Val(prod(size(w))))
end

position(w::Window) = w.position

# AbstractArray interface

Base.axes(::Window{<:Any,<:Any,X}) where X = X
Base.ndims(w::Window) = length(axes(w))
Base.size(w::Window) = length.(axes(w))

# TODO: we don't want a fully fledged OffsetArray, but having similar() and
#       copy() work would be nice
#Base.similar(w::Window, T::Type) = similar(w, T, size(w))

@inline function Base.getindex(w::Window{<:Any,N}, wi::Vararg{Int,N}) where N
    #wis = CartesianIndices(axes(w))
    wi = CartesianIndex(wi)

    # TODO: handle boundary with nothing
    # index within window?
    @boundscheck checkbounds(w, wi)

    pi = position(w) + wi

    # translated index within parent array?
    # (only necessary if window was created improperly)
    @boundscheck checkbounds(Bool, CartesianIndices(w.parent_size), pi) ||
        throw(BoundsError(unsafe_wrap(Array, w.parent_ptr, w.parent_size), Tuple(pi)))

    # TODO: would like to use LinearIndices here, but it creates extra
    #       instructions, fix upstream?
    # pli = LinearIndices(w.parent_size)[pi]
    pli = _sub2ind(w.parent_size, Tuple(pi)...)

    return unsafe_load(w.parent_ptr, pli)
end

Base.setindex(w::Window, wi::Int...) = throw(ArgumentError("mutation unsupported"))

# TODO: StridedArray interface for interior windows
#       need restructuring since we need access to parent strides
#Base.strides(w::Window) = substrides(strides(a), map((c, x) -> c + first(x) : c + last(x), Tuple(w.center), axes(w)))
#Base.unsafe_convert(::Type{Ptr{T}}, ::Window{T}) = w.ptr + (firstindex(w) - 1) * sizeof(T)


@inline (k::Kernel)(w::Window) = k.wf(w)
