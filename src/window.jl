using Base: @propagate_inbounds, _sub2ind, substrides

"""
    Window{X}(a::DenseArray, pos::CartesianIndex)

Create a stack-allocated view on `a` with axes `X` and cartesian indexing
relative to `pos` in the parent array.

The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
function Window end

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
    position(w::Window)::CartesianIndex

Return the position of `w` (i.e. its center coordinate) within its parent
array.
"""
position(w::Window) = w.position

# AbstractArray interface

Base.axes(::Window{<:Any,<:Any,X}) where X = X
Base.ndims(w::Window) = length(axes(w))
Base.size(w::Window) = length.(axes(w))

@inline function Base.getindex(w::Window{<:Any,N}, wi::Vararg{Int,N}) where N
    wi = CartesianIndex(wi)

    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds(Bool, w, wi) || return nothing

    pi = position(w) + wi

    # translated index within parent array?
    # (only necessary if window was created improperly)
    @boundscheck checkbounds(Bool, CartesianIndices(w.parent_size), pi) ||
        throw(BoundsError(unsafe_wrap(Array, w.parent_ptr, w.parent_size), (pi,)))

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
#
# TODO: we don't want a fully fledged OffsetArray, but having similar() and
#       copy() work would be nice
#Base.similar(w::Window, T::Type) = similar(w, T, size(w))

# FIXME: remove these as soon as we <:AbstractArray
Base.length(w::Window) = prod(size(w))
Base.keys(w::Window) = CartesianIndices(axes(w))
@propagate_inbounds Base.getindex(w::Window, i::CartesianIndex) =
    w[to_indices(w, (i,))...]
@inline Base.checkbounds(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, keys(w))
