using Test
using StaticFilters


a = rand(rand(5:10, 3)...)

dones = ntuple(i -> 1, ndims(a))
dzeros = ntuple(i -> 0, ndims(a))

for i in 1:ndims(a)
    ddir = ntuple(j -> Int(j == i), ndims(a))

    ksize = dones .+ Tuple(ddir)
    k = Kernel{ksize, dones}() do w
        return w[ddir...] - w[dzeros...]
    end

    @test isbits(k)

    x = map(k, a)
    y = diff(a, dims=i)

    @test x[axes(a,k)...] == y
end
