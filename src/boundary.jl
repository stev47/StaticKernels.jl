@inline getindex_boundary(_, _, b::Boundary) =
    throw(ArgumentError("undefined boundary behaviour for boundary type $b"))

@inline getindex_boundary(_, _, b::BoundaryNothing) = nothing

@inline getindex_boundary(_, _, b::BoundaryConstant) = b.value

@inline function getindex_boundary(w, wi, b::BoundaryReplicate)
    pi = position(w) + wi
    pimod = CartesianIndex(clamp.(Tuple(pi), ntuple(_->1, Val(length(wi))), w.parent_size))
    return getindex_parent(w, pimod)
end

@inline function getindex_boundary(w, wi, b::BoundaryCircular)
    pi = position(w) + wi
    pimod = CartesianIndex(mod1.(Tuple(pi), w.parent_size))
    return getindex_parent(w, pimod)
end

@inline function getindex_boundary(w, wi, b::BoundarySymmetric)
    pi = position(w) - wi
    pimod = CartesianIndex(map(mod1, Tuple(pi), w.parent_size))
    return getindex_parent(w, pimod)
end
