using Base: @propagate_inbounds, @_inline_meta

"""
    Window{X}(k::Kernel, a::DenseArray, pos::CartesianIndex)

Create a stack-allocated view on `a` with interior axes `X` and cartesian indexing
relative to `pos` in the parent array.
When indexing outside `X` the returned value is determined by the extension
specified in `k`.
The total axes of the window will span the full axes of kernel `k` except for
when `k` was created with an `ExtensionNothing` in which case they will be
trimmed.
"""
function Window end

# AbstractArray interface

Base.axes(w::Window) = extension(w.kernel) isa ExtensionNothing ? axes_inner(w) : axes(w.kernel)
Base.ndims(w::Window) = length(axes(w))
Base.size(w::Window) = length.(axes(w))

@inline checkbounds_inner(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, CartesianIndices(axes_inner(w)))

@propagate_inbounds Base.getindex(w::Window, wi::Int...) = getindex(w, CartesianIndex(wi))
@propagate_inbounds @inline function Base.getindex(w::Window{<:Any,N}, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds_inner(Bool, w, wi) || return getindex_extension(w, wi, extension(w.kernel))

    return parent(w)[position(w) + wi]
end

@propagate_inbounds Base.setindex!(w::Window, x, wi::Int...) = setindex!(w, x, CartesianIndex(wi))
@propagate_inbounds @inline function Base.setindex!(w::Window{<:Any,N}, x, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds_inner(Bool, w, wi) || return setindex_extension!(w, x, wi, extension(w.kernel))

    return parent(w)[position(w) + wi] = x
end

# Window interface

"""
    axes_inner(w::Window)

Return the inner axes of the window where no extension handling is active.
"""
axes_inner(w::Window{<:Any,<:Any,X}) where X = X

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
@generated function Base.Tuple(w::Window)
    # redefine it locally to avoid potential world-age issues
    _axes(::Type{W}) where W =
        W.parameters[4].parameters[3] == ExtensionNothing ?
            W.parameters[3] : W.parameters[4].parameters[1]

    return :( @_inline_meta; @inbounds ($((:(w[$i]) for i in CartesianIndices(_axes(w)))...),) )
end

"""
    parent(w::Window)

Return reference to parent array. This method may allocate.
"""
Base.parent(w::Window) = w.parent

# TODO: we don't want a fully fledged OffsetArray, but having similar() and
#       copy() work would be nice
#Base.similar(w::Window, T::Type) = similar(w, T, size(w))

# Workarounds

# FIXME: remove these as soon as we <:AbstractArray
Base.length(w::Window) = prod(size(w))
Base.keys(w::Window) = CartesianIndices(axes(w))
@inline Base.checkbounds(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, keys(w))
