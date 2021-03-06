using Base: @_inline_meta

"""
    windowloop(f, k::Kernel{kx}, a::AbstractArray)

Loops through `a` while calling `f(w)` at every index.
`f` is passed a kernel window `w` of size at most `kx`, potentially cropped
according to the range of `k`.

NOTE: It is assumed `kx` can fit inside `a`.
"""
@generated function windowloop(f, kernel::Kernel{kx}, acc, op, a::AbstractArray...) where {kx}
    # this assumes kx fits inside axes(x)
    wx(pos) = intersect.(kx, map((x,y) -> first(x) - y : last(x) - y, kx, pos))

    a1 = first(a)

    function loop_expr(pos)
        # current loop dimension
        d = length(kx) - length(pos)

        if d == 0
            ks = (Symbol('k', i) for i in eachindex(kx))
            ki = :( CartesianIndex($(ks...),) )
            ai = (:( Window{$(wx(pos))}(kernel, a[$i], $ki) ) for i in eachindex(a))
            return :( acc = op(acc, f( $(ai...) )) )
        end

        exprs = Expr[]
        k = Symbol('k', d)

        # lower boundary
        for i in first(kx[d]) : -1
            (a1 <: ExtensionArray && a1.parameters[4] != ExtensionNone) || break
            push!(exprs, :($k = first(axes(a1, $d)) + $(i - first(kx[d]))))
            push!(exprs, loop_expr((i, pos...)))
        end

        # interior
        push!(exprs, Expr(:for, :($k = ilo[$d] : iup[$d]), Expr(:block, loop_expr((0, pos...)))))

        # upper boundary
        for i in 1 : last(kx[d])
            (a1 <: ExtensionArray && a1.parameters[4] != ExtensionNone) || break
            push!(exprs, :($k = last(axes(a1, $d)) - $(last(kx[d]) - i)))
            push!(exprs, loop_expr((i, pos...)))
        end

        return Expr(:block, exprs...)
    end

    return quote
        Base.@_noinline_meta

        a1 = first(a)
        # lower and upper limits for interior
        ilo = first.(axes(a1)) .+ $(.-first.(kx))
        iup = last.(axes(a1)) .- $(last.(kx))

        $(loop_expr(()))

        return acc
    end
end
