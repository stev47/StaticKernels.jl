using Base: @_inline_meta

"""
    windowloop(f, k::Kernel{kx}, a::AbstractArray)

Loops through `a` while calling `f(w)` at every index.
`f` is passed a kernel window `w` of size at most `kx`, potentially cropped
according to the boundary handling of `k`.

NOTE: It is assumed `kx` can fit inside `a`.
"""
@generated function windowloop(
        f, kernel::Kernel{kx,<:Any,boundary}, a::AbstractArray) where {kx, boundary}
    # this assumes kx fits inside axes(x)
    wx(pos) = intersect.(kx, map((x,y) -> first(x) - y : last(x) - y, kx, pos))

    function loop_expr(pos)
        # current loop dimension
        d = length(kx) - length(pos)

        if d == 0
            ks = (Symbol('k', i) for i in eachindex(kx))
            ki = :( CartesianIndex($(ks...),) )
            return :( f( Window{$(wx(pos))}(kernel, a, $ki) ) )
        end

        exprs = Expr[]
        k = Symbol('k', d)

        # lower boundary
        for i in first(kx[d]) : -1
            boundary == BoundaryNone && break
            push!(exprs, :($k = first(axes(a, $d)) + $(i - first(kx[d]))))
            push!(exprs, loop_expr((i, pos...)))
        end

        # interior
        push!(exprs, Expr(:for,
            # FIXME: julia bug, "$k in ..." breaks generated function
            :($k = first(axes(a, $d)) + $(max(0, -first(kx[d]))) : last(axes(a, $d)) - $(max(0, last(kx[d])))),
            Expr(:block, loop_expr((0, pos...)))))

        # upper boundary
        for i in 1 : last(kx[d])
            boundary == BoundaryNone && break
            push!(exprs, :($k = last(axes(a, $d)) - $(last(kx[d]) - i)))
            push!(exprs, loop_expr((i, pos...)))
        end

        return Expr(:block, exprs...)
    end

    return quote
        # prevents allocation if f is mutating data in the caller scope
        @_inline_meta
        GC.@preserve a $(loop_expr(()))
        return nothing
    end
end
