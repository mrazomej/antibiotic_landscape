module Antibiotic

module viz
include("viz.jl")
end # submodule

module mh
include("metropolis.jl")
end # submodule

module geometry
include("geometry.jl")
end # submodule

module stats
include("stats.jl")
end # submodule

end # module Antibiotic
