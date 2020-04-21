"""
    getindex_extension(w::Window, wi::CartesianCoordinate, ext::Extension)

Return a value for an indexing operation on `w` with relative index `wi`
respecting the extension behaviour `ext`.
Users may define their own extension behaviour by defining additional methods
for this function.
"""
@inline getindex_extension(_, _, ext::Extension) =
    throw(ArgumentError("getindex_extension() undefined for extension type $ext"))

"""
    setindex_extension!(w::Window, x, wi::CartesianCoordinate, ext::Extension)

Store the value `x` for an indexing operation on `w` with relative index `wi`
respecting the extension behaviour `ext`.
Users may define their own extension behaviour by defining additional methods
for this function.
Depending on the extension a call to this function may be a no-op.
"""
@inline setindex_extension!(_, _, _, ext::Extension) =
    throw(ArgumentError("setindex_extension() undefined for extension type $ext"))

@inline getindex_extension(_, _, ext::ExtensionNothing) = nothing
@inline setindex_extension!(_, _, _, ext::ExtensionNothing) = nothing

@inline getindex_extension(_, _, ext::ExtensionConstant) = ext.value
@inline setindex_extension!(_, _, _, ext::ExtensionConstant) = ext.value

@inline function getindex_extension(w, wi, ext::ExtensionReplicate)
    pi = position(w) + wi
    pimod = CartesianIndex(clamp.(Tuple(pi), ntuple(_->1, Val(length(wi))), w.parent_size))
    return getindex_parent(w, pimod)
end
@inline function setindex_extension!(w, x, wi, ext::ExtensionReplicate)
    pi = position(w) + wi
    pimod = CartesianIndex(clamp.(Tuple(pi), ntuple(_->1, Val(length(wi))), w.parent_size))
    return setindex_parent!(w, x, pimod)
end

@inline function getindex_extension(w, wi, ext::ExtensionCircular)
    pi = position(w) + wi
    pimod = CartesianIndex(mod1.(Tuple(pi), w.parent_size))
    return getindex_parent(w, pimod)
end
@inline function setindex_extension!(w, x, wi, ext::ExtensionCircular)
    pi = position(w) + wi
    pimod = CartesianIndex(mod1.(Tuple(pi), w.parent_size))
    return setindex_parent!(w, x, pimod)
end

@inline function getindex_extension(w, wi, ext::ExtensionSymmetric)
    pi = position(w) - wi
    pimod = CartesianIndex(mod1.(Tuple(pi), w.parent_size))
    return getindex_parent(w, pimod)
end
@inline function setindex_extension!(w, x, wi, ext::ExtensionSymmetric)
    pi = position(w) - wi
    pimod = CartesianIndex(mod1.(Tuple(pi), w.parent_size))
    return setindex_parent!(w, x, pimod)
end
