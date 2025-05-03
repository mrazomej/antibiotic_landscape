## =============================================================================
println("Loading packages...")

# Import project package
import AutoEncoderToolkit

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

# Define dimensionality of latent space
n_latent = 2
# Define number of neurons in hidden layers
n_neuron = 128

# Define RHVAE hyper-parameters
T = 0.5f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 256 # Number of centroids

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/output/barbay_kinsler_2020/"

# Load data
df_kinsler = CSV.read(
    "$(data_dir)/kinsler_combined_hyperfitness.csv",
    DF.DataFrame
)

# Define number of environmenst
n_env = length(unique(df_kinsler.env))

# Pivot to extract standardized mean fitness values
df_kinsler_mean = DF.unstack(df_kinsler, :env, :id, :fitness_mean_standard)

# Extract fitness matrix
fitness_matrix = Float32.(Matrix(df_kinsler_mean[:, DF.Not(:env)]))

## =============================================================================

# Define path directory
path_dir = pwd()

# Select centroids via k-means
centroids_data = AutoEncoderToolkit.utils.centroids_kmedoids(
    fitness_matrix, n_centroids
)

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Generate output directory if it doesn't exist
if !isdir(out_dir)
    println("Generating output directory...")
    mkpath(out_dir)
end

## =============================================================================

println("Defining RHVAE architecture")

println("Define JointGaussianLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env => n_neuron, Flux.identity),
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
encoder = AutoEncoderToolkit.JointGaussianLogEncoder(
    encoder_chain, µ_layer, logσ_layer
)

## =============================================================================

println("Define SimpleGaussianDecoder...")
# Initialize decoder
decoder = AutoEncoderToolkit.SimpleGaussianDecoder(
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
        Flux.Dense(n_neuron => n_env, Flux.identity)
    )
)

## =============================================================================

println("Define MetricChain...")

# Define mlp chain
mlp_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env => n_neuron, Flux.identity),
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
metric_chain = AutoEncoderToolkit.RHVAEs.MetricChain(mlp_chain, diag, lower)

## =============================================================================

# Initialize rhvae
rhvae = AutoEncoderToolkit.RHVAEs.RHVAE(
    encoder * decoder,
    metric_chain,
    centroids_data,
    T,
    λ
)

## =============================================================================

println("Save model object...")

# Save model object
JLD2.save(
    "$(out_dir)/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)
