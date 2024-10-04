## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncode
import AutoEncode.diffgeo.NeuralGeodesics as NG

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import Random
Random.seed!(42)

## =============================================================================

# Define number of neurons in hidden layers
n_neuron = 32

## =============================================================================

println("Loading reference RHVAE model...")

# Load model
rhvae_template = JLD2.load("./output/gpu_model.jld2")["model"]

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
z_init, z_end = Float32.([0, 0]), Float32.([1, 1])

# Define NeuralGeodesic
nng = NG.NeuralGeodesic(mlp_chain, z_init, z_end)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/geodesic.jld2",
    Dict("model" => nng, "model_state" => Flux.state(nng))
)