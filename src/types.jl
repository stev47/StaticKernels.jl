abstract type Extension end

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

    """
        Kernel{X}(f)

    Create a kernel with axes `X` wrapping a kernel function `f`.

    The kernel function `f` defines a reduction of values within an `X`-sized
    view. When the kernel is applied to data of an array `a` the kernel function
    gets called with one argument `w::Window` that provides a local view on the
    data.

    ```@example
    # Laplacian 3x3 Kernel (i.e. axes (-1:1, -1:1))
    function f(w)
        return w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1]
    end
    Kernel{(-1:1,-1:1)}(f)
    ```

    For best performance you should annotate the kernel function `f` with `@inline`
    and index accesses within using `@inbounds`.
    """
    function Kernel{X}(f::F) where {X,F<:Function}
        X isa NTuple{<:Any,UnitRange{Int}} ||
            throw(ArgumentError("invalid axes: $X"))
        return new{X,F}(f)
    end
end

"""
    Window{T,N,X,K,A}

A stack-allocated array view fit for a kernel type `K` with indexing on axes
`X` relative to some position in the parent array of type `A`.
"""
# FIXME: inheriting from AbstractArray prevents us from constant-propagating
# getindex(w, i). See also https://github.com/JuliaLang/julia/issues/35531
#
#struct Window{T,N,X,K,A} <: AbstractArray{T,N}
struct Window{T,N,X,K,A}
    position::CartesianIndex{N}
    parent::A
    kernel::K

    """
        Window{X}(k::Kernel, a::AbstractArray, pos::CartesianIndex)

    Create a stack-allocated view on `a` with interior axes `X` and cartesian indexing
    relative to `pos` in the parent array.
    When indexing outside `X` the returned value is determined by
    [`StaticKernels.getindex_extension`](@ref).
    """
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
