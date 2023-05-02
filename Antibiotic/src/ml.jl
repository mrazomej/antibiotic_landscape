# Import functions related to vanilla autoencoders
module AEs
include("ae.jl")
end # sub-sub-module

##

# Import functions related to implicit rank minimization autoencoders
module IRMAEs
include("irmae.jl")
end # sub-sub-module

##

# Import functions related to implicit rank minimization autoencoders
module VAEs
include("vae.jl")
end # sub-sub-module


