using Base: require_one_based_indexing, _sub2ind, substrides

"""
    Window{T,N,K} <: AbstractArray{T,N}

A stack-allocated view with cartesian indexing relative to some center.
The user is responsible for ensuring that the parent array outlives this object
by using e.g. `GC.@preserve`.
"""
struct Window{T,N,K} <: AbstractArray{T,N}
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

center(w::Window) = w.center

# AbstractArray interface

Base.size(w::Window) = size(w.kernel)
Base.axes(w::Window) = axes(w.kernel)
# TODO: we don't want a fully fledged OffsetArray, but having similar() and
#       copy() work would be nice
#Base.similar(w::Window, T::Type) = similar(w, T, size(w))

# FIXME: only to support e.g. Base.last(), fix upstream?
Base.getindex(w::Window, wi::Int) = w[eachindex(w)[wi]]

@inline function Base.getindex(w::Window, wi::Int...)
    wi = CartesianIndex(wi)

    # index within window?
    @boundscheck checkbounds(Bool, LinearIndices(size(w.kernel)),
                             CartesianIndex(center(w.kernel)) + wi) ||
        throw(BoundsError(w, (wi,)))

    ci = center(w) + wi

    # translated index within parent array?
    # (only necessary if window was created improperly)
    @boundscheck checkbounds(LinearIndices(w.psize), ci)

    # TODO: would like to use LinearIndices here, but it creates extra
    #       instructions, fix upstream?
    # li = LinearIndices(w.psize)[ci]
    li = _sub2ind(w.psize, Tuple(ci)...)

    return unsafe_load(w.ptr, li)
end

Base.setindex(w::Window, wi::Int...) = throw(ArgumentError("mutation unsupported"))

# TODO: StridedArray interface for interior windows
#       need restructuring since we need access to parent strides
#Base.strides(w::Window) = substrides(strides(a), map((c, x) -> c + first(x) : c + last(x), Tuple(w.center), axes(w)))
#Base.unsafe_convert(::Type{Ptr{T}}, ::Window{T}) = w.ptr + (firstindex(w) - 1) * sizeof(T)


@inline @propagate_inbounds (k::Kernel)(w::Window) = k.f(w)
