# StaticKernels

Julia-native non-allocating kernel operations on arrays.
Current features include

- custom kernel functions in arbitrary dimensions
- custom boundary handling
- filter application through `map(kernel, array)`
- package is small and dependency free

## Usage

```julia
using StaticKernels
a = rand(1000, 1000)

# Laplace
k = Kernel{(-1:1,-1:1)}(w -> w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1])
map(k, a)

# Erosion
k = Kernel{(-1:1,-1:1)}(w -> minimum(Tuple(w)))
map(k, a)

# Laplace, zero boundary condition
k = Kernel{(-1:1,-1:1)}(w -> something(w[0,-1], 0.) + something(w[-1,0], 0.) - 4*w[0,0] + something(w[1,0], 0.) + something(w[0,1], 0.), StaticKernels.BoundaryNothing())
map(k, a)

# Forward-Gradient (non-skalar Kernel), neumann boundary condition
k = Kernel{(0:1, 0:1)}(w -> (something(w[1,0], w[0,0]) - w[0,0], something(w[0,1], w[0,0]) - w[0,0]), StaticKernels.BoundaryNothing())
map(k, a)
```

## User Notes

- you should use kernel functions declared with `@inline` and `@inbounds` for
  best performance.
- the package is aimed at small kernels, use different algorithms for larger
  kernels (inplace formulations or fft)
- (currently) high compilation time for larger kernels or higher dimensions for
  boundary specializations

## Implementation Notes

We use a statically sized array view `StaticKernels.Window` on which the
user-defined kernel function is applied. Access outside the window size returns
`nothing` instead of throwing an out-of-bounds error.

The user-supplied kernel function is specialized for all different `Windows`
(appropriately cropped versions on boundaries) and thus infers away
boundary-checks like `something(w[1,0], 0)` by leveraging constant propagation.

These components together with the auto-vectorizing Julia compiler allow for
fast execution.


## TODO

- generic mapreduce to allow e.g. `sum(k, a)`
- circular boundary conditions
- abstract/strided array interface for windows (blocked by julia issue)
- multi-window kernels for e.g. `map(k, a1, a2)`
- think about more specific kernel types and composability
- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(2,),(1,)}(w -> w[1] - w[0])`
