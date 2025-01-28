## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

# Import ML libraries
import Flux

# Import library to save models
import JLD2
import Glob

# Import basic math
import Random
Random.seed!(42)

## =============================================================================

# Define number of neurons in hidden layers
n_neuron = 32

## =============================================================================

# Locate current directory
path_dir = pwd()
# Find the path prefix where to store output
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match, "processing" => ""
)
# Define model directory
model_dir = "$(git_root())/output$(out_prefix)"

## =============================================================================

println("Loading reference RHVAE model...")

# Locate model file
file = first(Glob.glob("$(model_dir)/model.jld2"[2:end], "/"))
# Load model
rhvae_template = JLD2.load(file)["model"]

## =============================================================================

println("Defining NeuralGeodesic")

# Extract dimensionality of latent space
ldim = size(rhvae_template.centroids_latent, 1)

# Define mlp chain
mlp_chain = Flux.Chain(
    # First layer
    Flux.Dense(1 => n_neuron, Flux.identity),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.tanh_fast),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.tanh_fast),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.tanh_fast),
    # Output layer
    Flux.Dense(n_neuron => ldim, Flux.identity)
)

# Define initial and end points of the geodesic
z_init, z_end = Float32.([0, 0, 0]), Float32.([1, 1, 1])

# Define NeuralGeodesic
nng = NG.NeuralGeodesic(mlp_chain, z_init, z_end)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "$(model_dir)/geodesic.jld2",
    Dict("model" => nng, "model_state" => Flux.state(nng))
)