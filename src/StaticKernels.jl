module StaticKernels

export Kernel
export position

include("types.jl")

include("window.jl")
include("kernel.jl")

include("loop.jl")
include("operations.jl")

end
