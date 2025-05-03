## 
println("Loading packages...")

# Import project package
import AutoEncode
using AutoEncode.RHVAEs

# Import ML libraries
import Flux

# Import basic math
import Random
import LinearAlgebra
import StatsBase

# Import library to save model
import JLD2

Random.seed!(42)

## =============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 1_000

# Define number of neurons in non-linear hidden layers
n_neuron = 16
# Define dimensionality of latent space
n_latent = 2

## =============================================================================

println("Defining InfoMaxVAE architecture")

println("Define JointLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_input => n_neuron, Flux.identity),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.relu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.relu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.relu),
)

# Define layers for µ and log(σ)
µ_layer = Flux.Dense(n_neuron => n_latent, Flux.identity)
logσ_layer = Flux.Dense(n_neuron => n_latent, Flux.identity)

# build encoder
encoder = AutoEncode.JointLogEncoder(encoder_chain, µ_layer, logσ_layer)

## =============================================================================

println("Define SimpleDecoder...")
# Initialize decoder
decoder = AutoEncode.SimpleDecoder(
    Flux.Chain(
        # First layer
        Flux.Dense(n_latent => n_neuron, Flux.identity),
        # Second Layer
        Flux.Dense(n_neuron => n_neuron, Flux.relu),
        # Third layer
        Flux.Dense(n_neuron => n_neuron, Flux.relu),
        # Fourth layer
        Flux.Dense(n_neuron => n_neuron, Flux.relu),
        # Output layer
        Flux.Dense(n_neuron => n_input, Flux.identity)
    )
)

## =============================================================================

println("Define MutualInfoChain...")

# Define MLP layer and activation function
mlp_neurons = repeat([n_neuron], 3)
mlp_activation = repeat([Flux.relu], 3)

# Define MLP output activation function
mlp_output_activation = Flux.identity

mi = AutoEncode.InfoMaxVAEs.MutualInfoChain(
    n_input,
    n_latent,
    mlp_neurons,
    mlp_activation,
    mlp_output_activation
)

## =============================================================================

println("Initializing InfoMaxVAE...")

infomaxvae = AutoEncode.InfoMaxVAEs.InfoMaxVAE(
    encoder * decoder,
    mi
)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => infomaxvae, "model_state" => Flux.state(infomaxvae))
)