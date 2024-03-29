using Base: @_inline_meta

"""
    windowloop(f, k::Kernel{kx}, a::AbstractArray)

Loops through `a` while calling `f(w)` at every index.
`f` is passed a kernel window `w` of size at most `kx`, potentially cropped
according to the range of `k`.

NOTE: It is assumed `kx` can fit inside `a`.
"""
@generated function windowloop(f, kernel::Kernel{kx}, acc, op, a::AbstractArray...) where {kx}
    # calculate window interior axes
    # pos: offset at boundary, 0 for interior, ±n for lower/upper boundary
    function wx(pos)
        # this assumes kx fits inside axes(x)
        kx_shifted = map((x,y) -> first(x) - y : last(x) - y, kx, pos)
        return intersect.(kx, kx_shifted)
    end

    a1 = first(a)

    function loop_expr(pos)
        # current loop dimension
        d = length(kx) - length(pos)

        if d == 0
            ks = (Symbol('k', i) for i in eachindex(kx))
            ki = :( CartesianIndex($(ks...),) )
            ai = (:( Window{$(wx(pos))}(kernel, a[$i], $ki) ) for i in eachindex(a))
            return :( acc = @inline op(acc, f( $(ai...) )) )
        end

        exprs = Expr[]
        k = Symbol('k', d)

        # lower boundary
        for i in first(kx[d]) : -1
            has_extension(a1) || break
            push!(exprs, :($k = first(axes(a1, $d)) + $(i - first(kx[d]))))
            push!(exprs, loop_expr((i, pos...)))
        end

        # interior
        push!(exprs, Expr(:for, :($k = ilo[$d] : iup[$d]),
            Expr(:block, loop_expr((0, pos...)))))

        # upper boundary
        for i in 1 : last(kx[d])
            has_extension(a1) || break
            push!(exprs, :($k = last(axes(a1, $d)) - $(last(kx[d]) - i)))
            push!(exprs, loop_expr((i, pos...)))
        end

        return Expr(:block, exprs...)
    end

    # cutoff for interior limits
    kilo = has_extension(a1) ? min.(0, first.(kx)) : first.(kx)
    kiup = has_extension(a1) ? max.(0, last.(kx)) : last.(kx)

    return quote
        Base.@_noinline_meta

        a1 = first(a)
        # lower and upper limits for interior
        ilo = first.(axes(a1)) .- $(kilo)
        iup = last.(axes(a1)) .- $(kiup)

        $(loop_expr(()))

        return acc
    end
end
