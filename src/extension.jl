using Base: @propagate_inbounds

@inline index_extension(_, _, ext::Extension) =
    throw(ArgumentError("index_extension() undefined for extension $ext"))

@inline index_extension(a, i, ext::ExtensionReplicate) =
    CartesianIndex(clamp.(Tuple(i), 1, size(a)))

@inline index_extension(a, i, ext::ExtensionCircular) =
    CartesianIndex(mod1.(Tuple(i), size(a)))

symidx(k, n) = mod1(isodd(fld1(k, n)) ? k : -k, n)
@inline index_extension(a, i, ext::ExtensionSymmetric) =
    CartesianIndex(symidx.(Tuple(i), size(a)))

"""
    getindex_extension(w::Window, wi::CartesianCoordinate, ext::Extension)

Return a value for an indexing operation on `w` with relative index `wi`
respecting the extension behaviour `ext`.
Users may define their own extension behaviour by defining additional methods
for this function.
"""
getindex_extension

"""
    setindex_extension!(w::Window, x, wi::CartesianCoordinate, ext::Extension)

Store the value `x` for an indexing operation on `w` with relative index `wi`
respecting the extension behaviour `ext`.
Users may define their own extension behaviour by defining additional methods
for this function.
Depending on the extension a call to this function may be a no-op.
"""
setindex_extension!

@inline getindex_extension(_, _, ext::ExtensionNothing) = nothing
@inline setindex_extension!(_, _, _, ext::ExtensionNothing) = nothing

@inline getindex_extension(_, _, ext::ExtensionConstant) = ext.value
@inline setindex_extension!(_, _, _, ext::ExtensionConstant) = ext.value

@propagate_inbounds @inline function getindex_extension(w, wi, ext)
    a = parent(w)
    return a[index_extension(a, position(w) + wi, ext)]
end
@propagate_inbounds @inline function setindex_extension!(w, x, wi, ext)
    a = parent(w)
    return a[index_extension(a, position(w) + wi, ext)] = x
end
