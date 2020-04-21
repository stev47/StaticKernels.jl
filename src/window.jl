using Base: @propagate_inbounds, _sub2ind

"""
    Window{X}(k::Kernel, a::DenseArray, pos::CartesianIndex)

Create a stack-allocated view on `a` with axes `X` and cartesian indexing
relative to `pos` in the parent array.
The window axes span at most the axes of kernel type `K` and behaviour when
indexing outside them is determined by the extension specified by `K`.

The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
function Window end

# AbstractArray interface

Base.axes(::Window{<:Any,<:Any,X}) where X = X
Base.ndims(w::Window) = length(axes(w))
Base.size(w::Window) = length.(axes(w))

@propagate_inbounds Base.getindex(w::Window, wi::Int...) = getindex(w, CartesianIndex(wi))
@propagate_inbounds @inline function Base.getindex(w::Window{<:Any,N}, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds(Bool, w, wi) || return getindex_extension(w, wi, extension(w.kernel))

    return getindex_parent(w, position(w) + wi)
end

@propagate_inbounds Base.setindex!(w::Window, x, wi::Int...) = setindex!(w, x, CartesianIndex(wi))
@propagate_inbounds @inline function Base.setindex!(w::Window{<:Any,N}, x, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds(Bool, w, wi) || return setindex_extension!(w, x, wi, extension(w.kernel))

    return setindex_parent!(w, x, position(w) + wi)
end

# Window interface

"""
    position(w::Window)::CartesianIndex

Return the position of `w`, i.e. its center coordinate within the parent array.
"""
Base.position(w::Window) = w.position

"""
    Tuple(w::Window)

Create a tuple from the statically sized window `w`.

NOTE: this doesn't check bounds and thus assumes the window was properly
      created.
"""
@inline function Base.Tuple(w::Window{T}) where T
    ci = eachindex(w)
    @inline function f(i) @inbounds w[ci[i]]::T end
    return ntuple(f, Val(prod(size(w))))
end

"""
    getindex_parent(w::Window, i::CartesianIndex)

Equivalent to `parent(w)[i]` but non-allocating.
"""
@propagate_inbounds @inline function getindex_parent(w::Window{<:Any,N}, pi::CartesianIndex{N}) where N
    return unsafe_load(w.parent_ptr, _cart2lin(w, pi))
end

"""
    setindex_parent!(w::Window, x, i::CartesianIndex)

Equivalent to `parent(w)[i] = x` but non-allocating.
"""
@propagate_inbounds @inline function setindex_parent!(w::Window{<:Any,N}, x, pi::CartesianIndex{N}) where N
    return unsafe_store!(w.parent_ptr, x, _cart2lin(w, pi))
end

@inline function _cart2lin(w::Window, pi)
    @boundscheck checkbounds(Bool, CartesianIndices(w.parent_size), pi) ||
        throw(BoundsError(parent(w), (pi,)))

    # TODO: would like to use LinearIndices here, but it creates extra
    #       instructions, fix upstream?
    # return LinearIndices(w.parent_size)[pi]
    return _sub2ind(w.parent_size, Tuple(pi)...)
end

"""
    parent(w::Window)

Return reference to parent array. This method may allocate.
"""
Base.parent(w::Window) = unsafe_wrap(Array, w.parent_ptr, w.parent_size)

# TODO: we don't want a fully fledged OffsetArray, but having similar() and
#       copy() work would be nice
#Base.similar(w::Window, T::Type) = similar(w, T, size(w))

# Workarounds

# FIXME: remove these as soon as we <:AbstractArray
Base.length(w::Window) = prod(size(w))
Base.keys(w::Window) = CartesianIndices(axes(w))
@inline Base.checkbounds(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, keys(w))
