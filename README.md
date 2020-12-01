# StaticKernels

Julia-native non-allocating kernel operations on arrays.
Current features include

- custom kernel functions in arbitrary dimensions
- efficient custom boundary extensions
- kernel acts as a function in e.g. `map` or `mapreduce`
- package is small and dependency free

This package is not a big zoo of different kernels and filters, but instead
enables you to write them easily and efficiently yourself.

## Examples

```julia
using StaticKernels
a = rand(1000, 1000)

# 2d laplace finite difference
k = @kernel w -> w[0,-1] + w[-1,0] - 4*w[0,0] + w[1,0] + w[0,1]
map(k, a)
# using zero boundary condition
map(k, extend(a, StaticKernels.ExtensionConstant(0)))
# preallocated
b = similar(a, size(k, a))
map!(k, b, a)

# forward-gradient (non-scalar kernel), neumann boundary condition
k = @kernel w -> (w[1,0] - w[0,0], w[0,1] - w[0,0])
map(k, extend(a, StaticKernels.ExtensionReplicate()))

# custom boundary using `nothing`
k = @kernel w -> something(w[1,1], w[-1,-1], 0)
map(k, extend(a, StaticKernels.ExtensionNothing()))

# 2d total variation
k = @kernel w -> abs(w[1,0] - w[0,0]) + abs(w[0,1] - w[0,0])
sum(k, extend(a, StaticKernels.ExtensionReplicate()))

# erosion on a centered 3x3 window
k = Kernel{(-1:1,-1:1)}(@inline w -> minimum(w))
map(k, a)

# Conway's game of life
a = rand(Bool, 1000, 1000)
k = Kernel{(-1:1,-1:1)}(
    @inline w -> (count(w) - w[0,0] == 3 || count(w) == 3 && w[0,0]))
a .= map(k, extend(a, StaticKernels.ExtensionConstant(false)))
```

## User Notes

- for best performance you should annotate kernel functions with `@inline` and
  `@inbounds`
- the package is currently aimed at small-sized kernels, bigger-sized kernels
  have worse performance and you might want to consider using different
  algorithms (inplace reformulations or fft)
- (currently) high compilation time for larger kernels or boundary
  specialization in higher dimensions

## Implementation Notes

- a statically sized array view `StaticKernels.Window` with relative indexing
  is supplied to the user-defined kernel function
- the user-supplied kernel function is specialized for all cropped windows on
  the boundary, thus eliminating run-time checks.
- for cache efficiency boundaries are handled along with the interior instead
  of separately.
- for fully inlined kernel functions the Julia compiler manages to
  auto-vectorize the kernel loop efficiently in most cases.
