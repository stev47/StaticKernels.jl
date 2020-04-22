using Base: @_inline_meta

"""
    windowloop(f, k::Kernel{kx}, a::AbstractArray)

Loops through `a` while calling `f(w)` at every index.
`f` is passed a kernel window `w` of size at most `kx`, potentially cropped
according to the range of `k`.

NOTE: It is assumed `kx` can fit inside `a`.
"""
@generated function windowloop(f, kernel::Kernel{kx,<:Any,extension}, a::AbstractArray,
        acc, op) where {kx, extension}
    # this assumes kx fits inside axes(x)
    wx(pos) = intersect.(kx, map((x,y) -> first(x) - y : last(x) - y, kx, pos))

    function loop_expr(pos)
        # current loop dimension
        d = length(kx) - length(pos)

        if d == 0
            ks = (Symbol('k', i) for i in eachindex(kx))
            ki = :( CartesianIndex($(ks...),) )
            return :( acc = op(acc, f( Window{$(wx(pos))}(kernel, a, $ki) )) )
        end

        exprs = Expr[]
        k = Symbol('k', d)

        # lower boundary
        for i in first(kx[d]) : -1
            extension == ExtensionNone && break
            push!(exprs, :($k = first(axes(a, $d)) + $(i - first(kx[d]))))
            push!(exprs, loop_expr((i, pos...)))
        end

        # interior
        push!(exprs, Expr(:for, :($k = ilo[$d] : iup[$d]), Expr(:block, loop_expr((0, pos...)))))

        # upper boundary
        for i in 1 : last(kx[d])
            extension == ExtensionNone && break
            push!(exprs, :($k = last(axes(a, $d)) - $(last(kx[d]) - i)))
            push!(exprs, loop_expr((i, pos...)))
        end

        return Expr(:block, exprs...)
    end

    return quote
        # FIXME: inlining prevents allocation if f is mutating data in the
        # caller scope, but when accumulating it allocates ?!
        $(acc <: Nothing ? :(@_inline_meta) : :() )

        # lower and upper limits for interior
        ilo = first.(axes(a)) .+ $(max.(0, .-first.(kx)))
        iup = last.(axes(a)) .- $(max.(0, last.(kx)))

        $(loop_expr(()))

        return acc
    end
end
