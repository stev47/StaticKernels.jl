using Base: @propagate_inbounds

eltype_extension(a::AbstractArray) = eltype(a)

"""
    index_extension(a::AbstractArray, i::CartesianIndex, ext::Extension)

Return a valid index into `a` which corresponds to indexing `a` with `i`
respecting the extension `ext`.
Users may define their own extension behaviour by defining additional methods
for this function.
"""
index_extension

@inline index_extension(_, _, ext::Extension) =
    throw(ArgumentError("index_extension() undefined for extension $ext"))

_index_replicate(i, ax) = clamp(i, first(ax), last(ax))
@inline index_extension(a, i, ext::ExtensionReplicate) =
    CartesianIndex(_index_replicate.(Tuple(i), axes(a)))

_index_circular(i, ax) = mod(i - first(ax), length(ax)) + first(ax)
@inline index_extension(a, i, ext::ExtensionCircular) =
    CartesianIndex(_index_circular.(Tuple(i), axes(a)))

_index_symmetric(i, ax) =
    (j = i - first(ax); k = mod(j, length(ax));
        (iseven(fld(j, length(ax))) ? k : length(ax) - k - 1) + first(ax))
@inline index_extension(a, i, ext::ExtensionSymmetric) =
    CartesianIndex(_index_symmetric.(Tuple(i), axes(a)))

"""
    getindex_extension(a::AbstractArray, i::CartesianIndex, ext::Extension)

Retrieve a value at an out-of-bounds index `i` of array `a` with extension `ext`.
If no applicable method is implemented, [`StaticKernels.index_extension`](@ref)
will be used to determine an in-bounds index.
"""
getindex_extension

"""
    setindex_extension!(a::AbstractArray, x, i::CartesianIndex, ext::Extension)

Store the value `x` at an out-of-bounds index `i` of array `a` with extension `ext`.
If no applicable method is implemented, [`StaticKernels.index_extension`](@ref)
will be used to determine an in-bounds index.
Depending on the extension a call to this function may be non-sensical and thus
implemented as a no-op.
"""
setindex_extension!

# delegate from `Window` to parent `ExtensionArray`
@propagate_inbounds getindex_extension(w::Window, wi) =
    getindex_extension(parent(w), position(w) + wi)
@propagate_inbounds setindex_extension!(w::Window, x, wi) =
    setindex_extension!(parent(w), x, position(w) + wi)

# default implementation using `index_extension`
@propagate_inbounds getindex_extension(a, i, ext) =
    a[index_extension(a, i, ext)]
@propagate_inbounds setindex_extension!(a, x, i, ext) =
    a[index_extension(a, i, ext)] = x

@inline getindex_extension(_, _, ext::ExtensionNothing) = nothing
@inline setindex_extension!(_, _, _, ext::ExtensionNothing) = nothing

@inline getindex_extension(_, _, ext::ExtensionConstant) = ext.value
@inline setindex_extension!(_, _, _, ext::ExtensionConstant) = ext.value
