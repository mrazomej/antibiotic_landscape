## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncode

# Import libraries to handel data
import CSV
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

## =============================================================================

# Define number of hidden layers
n_hidden = 3
# Define number of neurons in non-linear hidden layers
n_neuron = 400
# Define dimensionality of latent space
n_latent = 2

## =============================================================================

println("Defining VAE architecture")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = [[Flux.identity]; repeat([Flux.relu], n_hidden - 1)]

## =============================================================================

println("Initializing vae...")

# Initialize encoder
encoder = AutoEncode.JointLogEncoder(
    n_env,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Initialize decoder
decoder = AutoEncode.SimpleDecoder(
    n_env,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize rhvae
vae = encoder * decoder

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => vae, "model_state" => Flux.state(vae))
)