module StaticKernels

export Kernel
export extend

include("types.jl")

include("extension.jl")
include("extendedarray.jl")
include("window.jl")
include("kernel.jl")

include("loop.jl")
include("operations.jl")

end
