using Base: @propagate_inbounds

# AbstractArray interface

Base.IndexStyle(a::ExtensionArray) = IndexCartesian()
Base.parent(a::ExtensionArray) = a.parent
Base.eltype(a::ExtensionArray) = eltype(parent(a))
Base.axes(a::ExtensionArray) = axes(parent(a))
Base.size(a::ExtensionArray) = size(parent(a))
Base.length(a::ExtensionArray) = length(parent(a))
Base.iterate(a::ExtensionArray, state...) = iterate(a, state...)
@propagate_inbounds Base.getindex(a::ExtensionArray, i...) =
    getindex(parent(a), i...)
@propagate_inbounds Base.setindex!(a::ExtensionArray, x, i...) =
    setindex!(parent(a), x, i...)

# ExtensionArray interface

extend(a::AbstractArray, extension::Extension) = ExtensionArray(a, extension)

extension(a::AbstractArray) = ExtensionNone()
extension(a::ExtensionArray) = a.extension

eltype_extension(a::ExtensionArray{<:Any,<:Any,<:Any,ExtensionConstant{S}}) where S = S

@propagate_inbounds getindex_extension(a::ExtensionArray, i) =
    getindex_extension(a, i, a.extension)
@propagate_inbounds setindex_extension!(a::ExtensionArray, x, i) =
    setindex_extension!(a, x, i, a.extension)
