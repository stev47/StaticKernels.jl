using Base: require_one_based_indexing

abstract type Extension end

struct ExtensionNone <: Extension end
struct ExtensionNothing <: Extension end
struct ExtensionReplicate <: Extension end
struct ExtensionCircular <: Extension end
struct ExtensionSymmetric <: Extension end
struct ExtensionConstant{T} <: Extension
    value::T
end

"""
    Kernel{X,F,Ext}

A kernel object with axes `X` that is wrapping a kernel function of type `F`
and specifying an extension `Ext`.
"""
struct Kernel{X,F,Ext}
    f::F
    extension::Ext

    function Kernel{X}(f::F, extension::Ext) where {X,Ext<:Extension,F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes: $X"))
        return new{X,F,Ext}(f, extension)
    end
end

"""
    Window{T,N,X,K}

A stack-allocated array view fit for a kernel type `K` with indexing on axes
`X` relative to some position in the parent array.
"""
# FIXME: inheriting from AbstractArray prevents us from constant-propagating
# getindex(w, i). See also https://github.com/JuliaLang/julia/pull/32105
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
