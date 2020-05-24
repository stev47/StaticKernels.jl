using Base: @propagate_inbounds

# AbstractArray interface

Base.IndexStyle(a::ExtensionArray) = IndexCartesian()
Base.parent(a::ExtensionArray) = a.parent
Base.eltype(a::ExtensionArray) = eltype(parent(a))
Base.axes(a::ExtensionArray) = map(r -> ExtensionUnitRange{UnboundedUnitRange()}(r), axes(parent(a)))
Base.size(a::ExtensionArray) = size(parent(a))
Base.length(a::ExtensionArray) = length(parent(a))
Base.iterate(a::ExtensionArray, state...) = iterate(parent(a), state...)
@propagate_inbounds Base.getindex(a::ExtensionArray, i...) =
    getindex(parent(a), i...)
@propagate_inbounds Base.setindex!(a::ExtensionArray, x, i...) =
    setindex!(parent(a), x, i...)

Base.similar(a::ExtensionArray, ::Type{T}, ax::NTuple{<:Any,<:ExtensionUnitRange}) where T =
    similar(a, T, parent.(ax))

Base.similar(::Type{A}, ax::NTuple{<:Any,<:ExtensionUnitRange}) where {A<:AbstractArray} =
    similar(A, parent.(ax))

function Base.copy!(dst::AbstractArray, src::ExtensionArray)
    all(isequal.(axes(src), axes(dst))) && return copy!(dst, parent(src))
    error("unimplemented")
end

# ExtensionArray interface

extend(a::AbstractArray, extension::Extension) = ExtensionArray(a, extension)

extension(a::AbstractArray) = ExtensionNone()
extension(a::ExtensionArray) = a.extension

eltype_extension(a::AbstractArray) = eltype(a)
eltype_extension(a::ExtensionArray{<:Any,<:Any,<:Any,ExtensionConstant{S}}) where S = S

@propagate_inbounds getindex_extension(a, i) =
    getindex_extension(a, i, a.extension)
@propagate_inbounds setindex_extension!(a, x, i) =
    setindex_extension!(a, x, i, a.extension)

#@propagate_inbounds function getindex_extension(a, i, j)
#    # this should be constant-propagated away
#    iszero(j) && return a[i]
#    return getindex_extension(a, i + j)
#end
#
#@propagate_inbounds function setindex_extension!(a, x, i, j)
#    # this should be constant-propagated away
#    iszero(j) && return a[i] = x
#    return setindex_extension!(a, x, i + j)
#end
