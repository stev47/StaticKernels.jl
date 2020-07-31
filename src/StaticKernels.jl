module StaticKernels

export Kernel
export extend
export @kernel

include("types.jl")

include("extension.jl")
include("extensionarray.jl")
include("window.jl")
include("kernel.jl")

include("loop.jl")
include("operations.jl")

end
