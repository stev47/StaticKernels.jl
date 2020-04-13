module StaticFilters

export Kernel
export position

include("types.jl")

include("window.jl")
include("windowloop.jl")
include("kernel.jl")

include("filters.jl")

end # module
