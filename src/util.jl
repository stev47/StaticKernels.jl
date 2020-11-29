@inline allequal(a) = isempty(a) ? true : mapfoldl(==(first(a)), &, a)

@inline function same(f, x...)
    fx = f.(x)
    allequal(fx) ||
        throw(DimensionMismatch("$f not equal: $(join(fx, " vs "))"))
    return first(fx)
end
