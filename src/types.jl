using Base: require_one_based_indexing

"""
    Kernel{X, F}

A kernel object with axes `X` that is wrapping a kernel function of type `F`.
"""
struct Kernel{X, F}
    f::F

    function Kernel{X}(f::F) where {X, F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes: $X"))
        return new{X, F}(f)
    end
end

"""
    Window{T,N,X}

A stack-allocated array view with indexing on axes `X` relative to some
position in the parent array.
"""
# FIXME: https://github.com/JuliaLang/julia/pull/32105
#struct Window{T,N,X} <: AbstractArray{T,N}
struct Window{T,N,X}
    position::CartesianIndex{N}
    parent_ptr::Ptr{T}
    parent_size::NTuple{N,Int}

    # TODO: relax to StridedArrays
    function Window{X}(a::DenseArray{T,N}, pos::CartesianIndex{N}) where {T,N,X}
        # because we use Base._sub2ind
        require_one_based_indexing(a)
        return new{T,N,X}(pos, pointer(a), size(a))
    end
end
