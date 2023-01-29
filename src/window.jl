using Base: @propagate_inbounds, @_inline_meta


# AbstractArray interface

Base.eltype(::Type{<:Window{T}}) where T = T
Base.eltype(w::Window) = eltype(typeof(w))
Base.axes(::Type{W}) where W<:Window =
    W.parameters[5] <: ExtensionArray &&
    W.parameters[5].parameters[4] != ExtensionNothing ?
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

# delegate extension to parent
@propagate_inbounds getindex_extension(w::Window, wi) =
    getindex_extension(parent(w), position(w) + wi)
@propagate_inbounds setindex_extension!(w::Window, x, wi) =
    setindex_extension!(parent(w), x, position(w) + wi)

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

## think twice before defining `iterate` like this, rather encourage use of
## mapreduce functions
#
#@generated function Base.iterate(w::Window, st = nothing)
#    ci = CartesianIndices(axes(w))
#    return quote
#        @_inline_meta
#        it = isnothing(st) ? iterate($ci) : iterate($ci, st)
#        isnothing(it) && return nothing
#        return @inbounds w[it[1]], it[2]
#    end
#end

# specialized mapfoldl, since Julia base has an upper limit on tuple size for
# inlining it

# since generated keyword functions may not inline, we forward to an internal
# generated non-keyword function
#   FIXME: file a julia bug
# this workaround does not seem to work when actually using non-default keyword
# arguments. explicitly specializing type arguments (F, Op) helps with the allocations
# but inlining still doesn't happen.
#   FIXME: file a julia bug

struct _DefaultValue end

@inline function Base.mapfoldl(f::F, op::Op, w::W...;
        dims = :, init = _DefaultValue()) where {F, Op, W<:Window}
    return _mapfoldl(f, op, dims, init, w...)
end

@generated function _mapfoldl(f, op, dims::Colon, init, w...)
    wax = same(axes, w...)
    weltype = same(eltype, w...)
    ws(i) = [ :( @inbounds w[$j][$i] ) for j in eachindex(w) ]
    stm_init = init === _DefaultValue ?
        :( acc = Base.mapreduce_empty(f, op, $weltype) ) :
        :( acc = init )
    stm_accs = [ :( acc = op(acc, f($(ws(i)...))) ) for i in CartesianIndices(wax) ]
    return :( @_inline_meta; $stm_init; $(stm_accs...); acc )
end

# Set of workarounds for not being able to subtype AbstractArray
#   see also `src/types.jl`
#   TODO: remove these as soon as Window <: AbstractArray

Base.ndims(w::Window) = length(axes(w))
Base.length(w::Window) = prod(size(w))
Base.CartesianIndices(w::Window) = CartesianIndices(axes(w))
Base.keys(w::Window) = CartesianIndices(w)
@inline Base.checkbounds(::Type{Bool}, w::Window, i::CartesianIndex) =
    in(i, keys(w))

# redefine inlined variations of various operations on arrays
# TODO: Julia base should define inlined dispatch function

@inline Base.mapreduce(f, op, w::Window...; dims = :, init = _DefaultValue()) =
    _mapfoldl(f, op, dims, init, w...)
@inline Base.in(x, w::Window) = _mapfoldl(==(x), |, :, _DefaultValue(), w)

@inline Base.sum(w::Window) = _mapfoldl(identity, +, :, _DefaultValue(), w)
@inline Base.sum(f, w::Window) = _mapfoldl(f, +, :, _DefaultValue(), w)
@inline Base.prod(w::Window) = _mapfoldl(identity, *, :, _DefaultValue(), w)
@inline Base.prod(f, w::Window) = _mapfoldl(f, *, :, _DefaultValue(), w)

# typemax/typemin usage here is not strictly compatible with julia base but
# simplifies the mapreduce implementation considerably
@inline Base.minimum(w::Window) = _mapfoldl(identity, min, :, typemax(eltype(w)), w)
@inline Base.minimum(f, w::Window) = _mapfoldl(f, min, :, typemax(eltype(w)), w)
@inline Base.maximum(w::Window) = _mapfoldl(identity, max, :, typemin(eltype(w)), w)
@inline Base.maximum(f, w::Window) = _mapfoldl(f, max, :, typemin(eltype(w)), w)

@inline Base.count(w::Window{Bool}) = _mapfoldl(identity, +, :, _DefaultValue(), w)
@inline Base.count(f, w::Window) = _mapfoldl(x->f(x)::Bool, +, :, _DefaultValue(), w)
@inline Base.all(w::Window{Bool}) = _mapfoldl(identity, &, :, _DefaultValue(), w)
@inline Base.all(f, w::Window) = _mapfoldl(x->f(x)::Bool, &, :, _DefaultValue(), w)
@inline Base.any(w::Window{Bool}) = _mapfoldl(identity, |, :, _DefaultValue(), w)
@inline Base.any(f, w::Window) = _mapfoldl(x->f(x)::Bool, |, :, _DefaultValue(), w)
