# StaticFilters

Faster, stack-allocated filter operations on arrays.

- supports arbitrary dimensions
- custom boundary handling
- linear filters with static kernel (finite difference operators)
- mapreduce filters (image morphology, etc.)
- filter composability
- package is small and dependency free

## Usage

```julia
using StaticFilters
a = rand(100, 100)
b = similar(a)

k = Kernel{(3,3)}(w -> w[0,-1] + w[-1,0] + 4*w[0,0] + w[1,0] + w[0,1])
k = Kernel{(3,3),(1,1)}(w -> w[0,-1] + w[-1,0] + 4*w[0,0] + w[1,0] + w[0,1])

map!(k, b, a)
map!(k, b, a1, a2)
```

## TODO

- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(2,),(1,)}(w -> w[1] - w[0])`
- boundary handling using `Union{T,Nothing}`
- vector simd performance `diff(v)`
- strided array interface for inner windows
