using Base.Broadcast: BroadcastStyle, Broadcasted, DefaultArrayStyle,
    broadcastable, instantiate, materialize!

"""
    UnboundedUnitRange{T} <: AbstractUnitRange{T}

Unbounded unit range representing all integers in type `T`.
"""
struct UnboundedUnitRange{T} <: AbstractUnitRange{T} end
UnboundedUnitRange() = UnboundedUnitRange{Int}()

Base.show(io::IO, ::MIME"text/plain", x::UnboundedUnitRange) = print(io, "ℤ")

Base.first(x::UnboundedUnitRange) = typemin(eltype(x))
Base.last(x::UnboundedUnitRange) = typemax(eltype(x))
Base.length(x::UnboundedUnitRange) = typemax(Int)

Base.intersect(x::AbstractUnitRange, y::UnboundedUnitRange) = intersect(y, x)
Base.intersect(x::UnboundedUnitRange, y::AbstractUnitRange) = y

"""
    ExtensionUnitRange{T,X,...} <: AbstractUnitRange{T}

Wraps an interior unit range, extending it with a statically known offset range
`X`.
"""
struct ExtensionUnitRange{T,X,IR<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
    interior::IR

    function ExtensionUnitRange{X}(interior::AbstractUnitRange{T}) where {T,X}
        isempty(X) && throw(ArgumentError("extension range is empty"))
        return new{T,X,typeof(interior)}(interior)
    end
end

const UnboundedExtensionUnitRange{T<:UnboundedUnitRange,X} =
    ExtensionUnitRange{T,X}

extension(x::ExtensionUnitRange{<:Any,X}) where X = X

Base.show(io::IO, mime::MIME"text/plain", x::ExtensionUnitRange) =
    (show(io, mime, parent(x)); print(io, " + "); show(io, mime, extension(x)))

Base.parent(x::ExtensionUnitRange) = x.interior

Base.first(x::ExtensionUnitRange) = first(parent(x))
Base.last(x::ExtensionUnitRange) = last(parent(x))

Base.intersect(x::ExtensionUnitRange, y::ExtensionUnitRange) =
    ExtensionUnitRange(intersect(x.inner, y.inner), intersect(x.outer, y.outer))


#@inline Base.checkbounds_indices(::Type{Bool}, IA::Tuple, I::Tuple, J::Tuple) =
#    checkindex(Bool, IA[1], I[1], J[1]) & checkbounds_indices(Bool, tail(IA), tail(I), tail(J))
#
#Base.@_propagate_inbounds_meta Base.checkindex(::Type{Bool}, IA::ExtensionUnitRange{<:Any,X}, I, J) where X =
#    checkindex(Bool, X, J) && checkindex(Bool, IA, I)


# N: dimensions
# X: relative axes of window that needs to be accessed

"""
    ExtensionStyle{N,X} <: Broadcast.AbstractArrayStyle{N}

Broadcasting style used for special extension indexing using
`getindex_extension(a, i, j)` using an interior index and a relative index `j`.
`X` denotes the allowable range for the relative index `j`.
"""
struct ExtensionStyle{N,X} <: Broadcast.AbstractArrayStyle{N} end

# TODO: mixed N, M
#
BroadcastStyle(::ExtensionStyle{N,X}, ::ExtensionStyle{M,Y}) where {N,M,X,Y} =
    BroadcastStyle(ExtensionStyle{max(N,M),X}, ExtensionStyle{max(N,M),Y})

BroadcastStyle(ks::ExtensionStyle{N,X}, ::DefaultArrayStyle{N}) where {N,X} = ks



#Base.size(bc::Broadcasted) = map(length, axes(bc))
Base.similar(bc::Broadcasted{<:ExtensionStyle{N,X}}, ::Type{T}) where {N,X,T} =
    similar(Array{T}, size(bc) .- length.(X) .+ 1)

# the default `materialize!` uses axes(dest), but we take care of axes in copyto!
@inline Broadcast.materialize!(::ExtensionStyle, dest, bc::Broadcasted) =
    copyto!(dest, instantiate(bc))

# julia 1.4 fix
@inline Broadcast.materialize!(dest, bc::Broadcasted{<:ExtensionStyle}) =
    materialize!(Broadcast.combine_styles(dest, bc), dest, bc)

Broadcast.instantiate(bc::Broadcasted{<:ExtensionStyle}) = bc

@inline function Base.copyto!(dst::AbstractArray, bc::Broadcasted{ExtensionStyle{N,X}}) where {N,X}
    size(dst) == size(bc) || throw(ArgumentError("sizes don't match: $(size(dst)) vs $(size(bc))"))

    o = CartesianIndex(first.(axes(dst)) .- first.(axes(bc)))
    for i in CartesianIndices(axes(bc))
        @inbounds dst[o + i] = bc[i]
    end

    return dst
end

struct ShiftedArray{T,N,S,P} <: AbstractArray{T,N}
    parent::P

    # TODO: assert on S
    ShiftedArray{S}(a::AbstractArray{T,N}) where {T,N,S} =
        new{T,N,S,typeof(a)}(a)
end

Base.parent(a::ShiftedArray) = a.parent

shift(::ShiftedArray{<:Any,<:Any,S}) where S = S

# AbstractArray
Base.size(a::ShiftedArray) = size(parent(a))
Base.@propagate_inbounds Base.getindex(a::ShiftedArray, i...) = a[CartesianIndex(i...)]

# Broadcast
BroadcastStyle(::Type{<:ShiftedArray{T,N,S}}) where {T,N,S} =
    ExtensionStyle{N,map(s -> -s:-s, S)}()

Base.axes(a::ShiftedArray) = map((ax,s) -> ax .+ s, axes(parent(a)), shift(a))
#Base.ndims(::Type{<:ShiftedArray{T,N}}) where {T,N} = N

Base.@propagate_inbounds Base.getindex(a::ShiftedArray, i::CartesianIndex) = parent(a)[i - CartesianIndex(shift(a))]
Broadcast.broadcastable(a::ShiftedArray) = a
