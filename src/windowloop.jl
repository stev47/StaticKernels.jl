using Base: @_inline_meta

"""
    windowloop(f, a::AbstractArray, Val(kx))

Loops through `a` while calling `f(w)` at every index.
`f` is passed a window `w` of size at most `kx`, cropped according to the
boundaries of `a` at the current index.

NOTE: It is assumed `kx` can fit inside `a`.
"""
@generated function windowloop(f, a::AbstractArray, ::Val{kx}) where kx
    # this assumes kx fits inside axes(x)
    wx(pos) = intersect.(kx, map((x,y) -> first(x) - y : last(x) - y, kx, pos))

    function loop_expr(pos)
        # current loop dimension
        d = length(kx) - length(pos)

        if d == 0
            ks = (Symbol('k', i) for i in eachindex(kx))
            ki = :( CartesianIndex($(ks...),) )
            return :( f( Window{$(wx(pos))}(a, $ki) ) )
        end

        exprs = Expr[]
        k = Symbol('k', d)

        # lower boundary
        for i in first(kx[d]) : -1
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
            push!(exprs, :($k = last(axes(a, $d)) - $(last(kx[d]) - i)))
            push!(exprs, loop_expr((i, pos...)))
        end

        return Expr(:block, exprs...)
    end

    return quote
        # prevents allocation if f is mutating data in the caller scope
        @_inline_meta
        $(loop_expr(()))
    end
end
