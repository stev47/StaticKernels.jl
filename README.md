# StaticKernels

Julia-native non-allocating kernel operations on arrays.
Current features include

- custom kernel functions in arbitrary dimensions
- custom boundary/extension handling
- kernel acts as a function in `map`, `map!`
- package is small and dependency free

## Usage

```julia
using StaticKernels
a = rand(1000, 1000)

# laplace
kf(w) = w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1]
k = Kernel{(-1:1,-1:1)}(kf)
map(k, a)

# erosion
k = Kernel{(-1:1,-1:1)}(w -> minimum(Tuple(w)))
map(k, a)

# laplace, zero boundary condition
k = Kernel{(-1:1,-1:1)}(kf, StaticKernels.ExtensionConstant(0))
map(k, a)

# forward-gradient (non-skalar kernel), neumann boundary condition
kf(w) = (w[1,0] - w[0,0], w[0,1] - w[0,0])
k = Kernel{(0:1, 0:1)}(kf, StaticKernels.ExtensionReplicate())
map(k, a)

# custom boundary using `nothing`
kf(w) = something(w[1,1], w[-1,-1], 0)
k = Kernel{(-1:1, -1:1)}(kf, StaticKernels.ExtensionNothing())
map(k, a)
```

## User Notes

- for best performance you should annotate kernel functions with `@inline` and
  `@inbounds`
- the package is aimed at small kernels, use different algorithms for larger
  kernels (inplace formulations or fft)
- (currently) high compilation time for larger kernels or higher dimensions for
  boundary specializations

## Implementation Notes

We use a statically sized array view `StaticKernels.Window` on which the
user-defined kernel function is applied. Access outside the window size returns
`nothing` instead of throwing an out-of-bounds error.

The user-supplied kernel function is specialized for all different `Windows`
(appropriately cropped versions on boundaries) and thus infers away checks like
e.g. `something(w[1,0], 0)` by leveraging constant propagation.

These components together with the auto-vectorizing Julia compiler allow for
fast execution.


## TODO

- generic mapreduce to allow e.g. `sum(k, a)`
- abstract/strided array interface for windows (blocked by julia issue)
- multi-window kernels for e.g. `map(k, a1, a2)`
- think about more specific kernel types and composability
- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(2,),(1,)}(w -> w[1] - w[0])`
