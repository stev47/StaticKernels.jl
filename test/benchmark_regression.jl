using BenchmarkTools, Printf
using StaticKernels

_bench_i = 1
if !@isdefined _bench_times
    _bench_times = Float64[]
end

amean(v) = mapreduce(x->x, +, v) / length(v)
gmean(v) = mapreduce(x->x^(1/length(v)), *, v)
hmean(v) = length(v) / mapreduce(x->1/x, +, v)

function showreg(trial, tol=0.05)
    global _bench_times, _bench_i

    while length(_bench_times) < _bench_i
        push!(_bench_times, 0)
    end

    old = _bench_times[_bench_i]
    new = minimum(trial.times)

    rel = (new - old) / old

    str = abs(rel) < tol ? "→" : new < old ? "↓" : "↑"

    @printf("  [%3d]: %10.0f %s %10.0f (%+1.3f)\n", _bench_i, round(old), str, round(new), rel)

    _bench_times[_bench_i] = new
    _bench_i += 1

    return nothing
end

macro mybench(x)
    b = esc(:(@benchmark($x, seconds=0.3)))
    return :(showreg($b))
end


# Benchmarks


function laplace(n)
    a = rand(n, n)
    k = Kernel{(-1:1,-1:1)}(@inline function f(w) @inbounds w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1] end)
    b = similar(a, size(k, a))

    @mybench map!($k, $b, $a)
end
println("laplace")
foreach(laplace, (1000, 3000))

a = rand(1000, 1000)
@inline kf(w) = @inbounds w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1]
k = Kernel{(-1:1,-1:1)}(kf, StaticKernels.ExtensionConstant(0))
b = similar(a, size(k, a))

@mybench map!($k, $b, $a)
