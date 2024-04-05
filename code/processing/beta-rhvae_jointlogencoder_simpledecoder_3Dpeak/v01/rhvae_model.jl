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

## =============================================================================

println("Defining RHVAE architecture")

println("Define JointLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_input => n_neuron, Flux.identity),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
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
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Third layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Fourth layer
        Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
        # Output layer
        Flux.Dense(n_neuron => n_input, Flux.identity)
    )
)

## =============================================================================

println("Define MetricChain...")

# Define mlp chain
mlp_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_input => n_neuron, Flux.identity),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
    # Fourth layer
    Flux.Dense(n_neuron => n_neuron, Flux.leakyrelu),
)

# Define layers for the diagonal and lower triangular part of the covariance
# matrix
diag = Flux.Dense(n_neuron => n_latent, Flux.identity)
lower = Flux.Dense(
    n_neuron => n_latent * (n_latent - 1) ÷ 2, Flux.identity
)

# Build metric chain
metric_chain = AutoEncode.RHVAEs.MetricChain(mlp_chain, diag, lower)


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