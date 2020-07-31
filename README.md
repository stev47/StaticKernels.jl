# StaticKernels

Julia-native non-allocating kernel operations on arrays.
Current features include

- custom kernel functions in arbitrary dimensions
- efficient custom boundary extensions
- kernel acts as a function in e.g. `map` or `mapreduce`
- package is small and dependency free

This package is not a big zoo of different kernels and filters, but instead
enables you to write them easily and efficiently yourself.

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
k = Kernel{(-1:1,-1:1)}(kf)
map(k, extend(a, StaticKernels.ExtensionConstant(0)))

# forward-gradient (non-scalar kernel), neumann boundary condition
kf(w) = (w[1,0] - w[0,0], w[0,1] - w[0,0])
k = Kernel{(0:1, 0:1)}(kf)
map(k, extend(a, StaticKernels.ExtensionReplicate()))

# custom boundary using `nothing`
kf(w) = something(w[1,1], w[-1,-1], 0)
k = Kernel{(-1:1, -1:1)}(kf)
map(k, extend(a, StaticKernels.ExtensionNothing()))

# total variation
kf(w) = abs(w[1,0] - w[0,0]) + abs(w[0,1] - w[0,0])
k = Kernel{(0:1,0:1)}(kf)
sum(k, extend(a, StaticKernels.ExtensionReplicate()))
```

## User Notes

- for best performance you should annotate kernel functions with `@inline` and
  `@inbounds`
- the package is currently aimed at small kernels, for bigger kernels consider
  using different algorithms (inplace formulations or fft)
- (currently) high compilation time for larger kernels or higher dimensions for
  boundary specializations

## Implementation Notes

- a statically sized array view `StaticKernels.Window` with relative indexing
  is supplied to the user-defined kernel function
- the user-supplied kernel function is specialized for all cropped windows on
  the boundary, thus eliminating run-time checks.
- for cache efficiency boundaries are handled along with the interior instead
  of separately.
- for fully inlined kernel functions the Julia compiler manages to
  auto-vectorize the kernel loop efficiently in most cases.

## TODO

- make `k(a)` create a broadcastable object (define broadcast style, use
  windowloop in `copy!(bc)` and define appropriate axes)
- introduce elementary kernel `shift(a, CartesianIndex(1, 1))`, `@shift a[1,1]`
- nicer (but type-instable) interface for kernel creation
- abstract/strided array interface for windows (blocked by julia issue)
- think about more specific kernel types and composability
- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(0:1,)}(w -> w[1] - w[0])`
