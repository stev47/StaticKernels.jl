module StaticKernels

export Kernel
export extend

include("types.jl")

include("extension.jl")
include("extensionarray.jl")
include("window.jl")
include("kernel.jl")

include("loop.jl")
include("operations.jl")

include("broadcast.jl")

end
