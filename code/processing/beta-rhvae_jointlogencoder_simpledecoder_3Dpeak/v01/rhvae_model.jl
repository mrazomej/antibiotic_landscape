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

## ============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 1_000

# Define number of samples in batch
n_batch = 128

# Define number of hidden layers
n_hidden = 3
# Define number of neurons in non-linear hidden layers
n_neuron = 32
# Define dimensionality of latent space
n_latent = 2

# Define RHVAE hyper-parameters
T = 0.8f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 256 # Number of centroids

## ============================================================================

println("Generating synthetic data...")

# Define function
f(x₁, x₂) = 10.0f0 * exp(-(x₁^2 + x₂^2))

# Defien radius
radius = 3

# Sample random radius
r_rand = radius .* sqrt.(Random.rand(n_data))

# Sample random angles
θ_rand = 2π .* Random.rand(n_data)

# Convert form polar to cartesian coordinates
x_rand = Float32.(r_rand .* cos.(θ_rand))
y_rand = Float32.(r_rand .* sin.(θ_rand))
# Feed numbers to function
z_rand = f.(x_rand, y_rand)

# Compile data into matrix
data = Matrix(hcat(x_rand, y_rand, z_rand)')

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=2)

# Center data to have mean zero and standard deviation one
data_std = StatsBase.transform(dt, data);

## ============================================================================

# Select centroids via k-means
centroids_data = AutoEncode.utils.centroids_kmedoids(data_std, n_centroids)

## ============================================================================

println("Defining VAE architecture")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.leakyrelu], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.leakyrelu], n_hidden)

# Define MetricChain layer and activation function
metric_neurons = repeat([n_neuron], n_hidden)
metric_activation = repeat([Flux.leakyrelu], n_hidden)

## ============================================================================

println("Initializing rhvae...")

# Initialize encoder
encoder = AutoEncode.JointLogEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Initialize decoder
decoder = AutoEncode.SimpleDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Define Metric MLP
metric_chain = AutoEncode.RHVAEs.MetricChain(
    n_input,
    n_latent,
    metric_neurons,
    metric_activation,
    Flux.identity
)


# Initialize rhvae
rhvae = AutoEncode.RHVAEs.RHVAE(
    encoder * decoder,
    metric_chain,
    centroids_data,
    T,
    λ
)

## ============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "./output/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)