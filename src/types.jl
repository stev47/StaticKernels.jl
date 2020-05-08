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
    Kernel{X,F}

A kernel object with axes `X` that is wrapping a kernel function of type `F`.
"""
struct Kernel{X,F}
    f::F

    function Kernel{X}(f::F) where {X,F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes: $X"))
        return new{X,F}(f)
    end
end

"""
    Window{T,N,X,K,A}

A stack-allocated array view fit for a kernel type `K` with indexing on axes
`X` relative to some position in the parent array.
"""
# FIXME: inheriting from AbstractArray prevents us from constant-propagating
# getindex(w, i). See also https://github.com/JuliaLang/julia/issues/35531
#
#struct Window{T,N,X,K,A} <: AbstractArray{T,N}
struct Window{T,N,X,K,A}
    position::CartesianIndex{N}
    parent::A
    kernel::K

    function Window{X}(k::Kernel, a::AbstractArray{T,N}, pos::CartesianIndex{N}) where {T,N,X}
        return new{T,N,X,typeof(k),typeof(a)}(pos, a, k)
    end
end

"""
    ExtensionArray{T,N,A,Ext} <: AbstractArray{T,N}

An array-wrapper mimicking its parent but additonally specifying an extension
for determining how to index out-of-bounds.
"""
struct ExtensionArray{T,N,A,Ext} <: AbstractArray{T,N}
    parent::A
    extension::Ext

    function ExtensionArray(parent::AbstractArray, extension::Extension)
        return new{eltype(parent),ndims(parent),typeof(parent),typeof(extension)}(parent, extension)
    end
end
