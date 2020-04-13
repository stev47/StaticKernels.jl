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

# Laplace
k = Kernel{(-1:1,-1:1)}(w -> w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1])
map(k, a)
```


## Examples


### Non-scalar Output

### Fast Local Matrix Operations

```julia
using StaticArrays, LinearAlgebra

@inline function(w)
    m = SMatrix{size(w)...}(Tuple(w))
    return det(m)
end

a = rand(100, 100)
map(Kernel{(3,3)}(wf), a)
```



## TODO

- generic mapreduce
- circular boundary conditions
- abstract array interface for windows
- multi-window kernels
- strided array interface for windows
- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(2,),(1,)}(w -> w[1] - w[0])`
