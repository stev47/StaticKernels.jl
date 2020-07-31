## TODO

- make `k(a)` create a broadcastable object (define broadcast style, use
  windowloop in `copy!(bc)` and define appropriate axes)
- introduce elementary kernel `shift(a, CartesianIndex(1, 1))`, `@shift a[1,1]`
- nicer (but type-instable) interface for kernel creation
- abstract/strided array interface for windows (blocked by julia issue)
- think about more specific kernel types and composability
- syntactic sugar for determining kernel size through index access:
  `@kernel(w -> w[1] - w[0]) == Kernel{(0:1,)}(w -> w[1] - w[0])`
