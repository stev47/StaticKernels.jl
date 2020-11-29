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

Base.axes(::Type{W}) where W<:Window =
    W.parameters[5] <: ExtensionArray &&
    W.parameters[5].parameters[4] != ExtensionNothing &&
    W.parameters[5].parameters[4] != ExtensionNone ?
        axes(W.parameters[4]) :
        W.parameters[3]

Base.axes(w::Window) = axes(typeof(w))
Base.size(w::Window) = length.(axes(w))

@propagate_inbounds Base.getindex(w::Window, wi::Int...) = getindex(w, CartesianIndex(wi))
@propagate_inbounds function Base.getindex(w::Window{<:Any,N}, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds_inner(Bool, w, wi) || return getindex_extension(w, wi)

    return parent(w)[position(w) + wi]
end

@propagate_inbounds Base.setindex!(w::Window, x, wi::Int...) = setindex!(w, x, CartesianIndex(wi))
@propagate_inbounds function Base.setindex!(w::Window{<:Any,N}, x, wi::CartesianIndex{N}) where N
    # central piece to get efficient boundary handling.
    # we rely on the compiler to constant propagate this check away
    checkbounds_inner(Bool, w, wi) || return setindex_extension!(w, x, wi)

    return parent(w)[position(w) + wi] = x
end


# Window interface

Base.eltype(::Type{<:Window{T}}) where T = T

"""
    checkbounds_inner(Bool, w::Window, i::CartesianIndex)

Return true if `i` indexes `w`'s parent in the interior and false if an
extension would be involved.
"""
@inline checkbounds_inner(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, CartesianIndices(axes_inner(w)))

"""
    parent(w::Window)

Return reference to parent array.
"""
Base.parent(w::Window) = w.parent

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
    els = [ :( w[$i] ) for i in CartesianIndices(axes(w)) ]
    return :( @_inline_meta; @inbounds ($(els...),) )
end


# Set of workarounds for not being able to subtype AbstractArray
#   see also `src/types.jl`
#   TODO: remove these as soon as Window <: AbstractArray

Base.ndims(w::Window) = length(axes(w))
Base.eltype(w::Window) = eltype(typeof(w))
Base.length(w::Window) = prod(size(w))
Base.CartesianIndices(w::Window) = CartesianIndices(axes(w))
Base.keys(w::Window) = CartesianIndices(w)
@inline Base.checkbounds(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, keys(w))
