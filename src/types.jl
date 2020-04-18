using Base: require_one_based_indexing

abstract type Boundary end

struct BoundaryNone <: Boundary end
struct BoundaryNothing <: Boundary end
struct BoundaryReplicate <: Boundary end
struct BoundaryCircular <: Boundary end
struct BoundarySymmetric <: Boundary end
struct BoundaryConstant{T} <: Boundary
    value::T
end

"""
    Kernel{X,F,B}

A kernel object with axes `X` that is wrapping a kernel function of type `F`
and specifying boundary conditions `B`.
"""
struct Kernel{X,F,B}
    f::F
    boundary::B

    function Kernel{X}(f::F, boundary::B) where {X,B<:Boundary,F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes: $X"))
        return new{X,F,B}(f, boundary)
    end
end

"""
    Window{T,N,X,K}

A stack-allocated array view fit for a kernel type `K` with indexing on axes
`X` relative to some position in the parent array.
"""
# FIXME: https://github.com/JuliaLang/julia/pull/32105
#struct Window{T,N,K} <: AbstractArray{T,N}
struct Window{T,N,X,K}
    position::CartesianIndex{N}
    parent_ptr::Ptr{T}
    parent_size::NTuple{N,Int}
    kernel::K

    # TODO: relax to StridedArrays
    function Window{X}(k::Kernel, a::DenseArray{T,N}, pos::CartesianIndex{N}) where {T,N,X}
        # because we use Base._sub2ind
        require_one_based_indexing(a)
        return new{T,N,X,typeof(k)}(pos, pointer(a), size(a), k)
    end
end
