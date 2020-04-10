using Base: require_one_based_indexing, _sub2ind

"""
    Window{T,N}

A stack-allocated view with cartesian indexing relative to some center.
The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
struct Window{T,N}
    ptr::Ptr{T}
    center::CartesianIndex{N}
    psize::NTuple{N,Int}

    function Window(a::DenseArray{T,N}, center::CartesianIndex{N}) where {T,N}
        require_one_based_indexing(a)
        ptr = pointer(a)
        psize = size(a)
        return new{T,N}(ptr, center, psize)
    end
end

Base.eltype(w::Window{T}) where T = T
Base.ndims(w::Window{<:Any,N}) where N = N

@inline function Base.getindex(w::Window, wi...)
    ci = w.center + CartesianIndex(wi)

    @boundscheck checkbounds(LinearIndices(w.psize), ci)

    # TODO: would like to use LinearIndices here, but it creates extra
    #       instructions
    # li = LinearIndices(w.psize)[ci]
    li = _sub2ind(w.psize, Tuple(ci)...)
    return unsafe_load(w.ptr, li)
end
