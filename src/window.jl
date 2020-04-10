using Base: require_one_based_indexing, _sub2ind

"""
    Window{T,N}

A stack-allocated view with cartesian indexing relative to some center.
The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
struct Window{T,N,K}
    ptr::Ptr{T}
    center::CartesianIndex{N}
    psize::NTuple{N,Int}
    kernel::K

    function Window(a::DenseArray{T,N}, k::Kernel, center::CartesianIndex{N}) where {T,N}
        ndims(k) == ndims(a) ||
            throw(ArgumentError("invalid number of dimensions: $(ndims(k)) vs $(ndims(a))"))
        require_one_based_indexing(a)
        ptr = pointer(a)
        psize = size(a)
        return new{T,N,typeof(k)}(ptr, center, psize, k)
    end
end

Base.eltype(w::Window{T}) where T = T
Base.ndims(w::Window{<:Any,N}) where N = N
center(w::Window) = w.center

@inline function Base.getindex(w::Window, wi...)
    wi = CartesianIndex(wi)

    # within window?
    @boundscheck checkbounds(Bool, LinearIndices(size(w.kernel)),
                             CartesianIndex(center(w.kernel)) + wi) ||
        throw(BoundsError(w, wi))

    ci = center(w) + wi

    # within parent array?
    # (only necessary if window was created improperly)
    @boundscheck checkbounds(LinearIndices(w.psize), ci)

    # TODO: would like to use LinearIndices here, but it creates extra
    #       instructions, fix in base?
    # li = LinearIndices(w.psize)[ci]
    li = _sub2ind(w.psize, Tuple(ci)...)

    return unsafe_load(w.ptr, li)
end

@inline @propagate_inbounds (k::Kernel)(w::Window) = k.f(w)
