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

# Define dimensionality of latent space
n_latent = 2
# Define number of neurons in hidden layers
n_neuron = 400

# Define RHVAE hyper-parameters
T = 0.8f0 # Temperature
λ = 1.0f-2 # Regularization parameter
n_centroids = 64 # Number of centroids

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/data/Maeda_2020"

# Load file into memory
df_res = CSV.read("$(data_dir)/maeda_resistance_tidy.csv", DF.DataFrame)

# Initialize array to save data
ic50_mat = Matrix{Float32}(
    undef, length(unique(df_res.stress)), length(unique(df_res.strain))
)
# Group data by strain
df_group = DF.groupby(df_res, :strain, sort=true)

# Extract unique stresses to make sure the matrix is built correctly
stress = sort(unique(df_res.stress))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :stress)
    # Check that the stress are in the correct order
    if all(data.stress .== stress)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)
# Define number of samples
n_samples = size(ic50_mat, 2)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat)

## =============================================================================

# Select centroids via k-means
centroids_data = AutoEncode.utils.centroids_kmedoids(ic50_std, n_centroids)

## =============================================================================

println("Defining RHVAE architecture")

println("Define JointLogEncoder...")
# Define encoder chain
encoder_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env => n_neuron, Flux.identity),
    # Add normalization layer
    Flux.BatchNorm(n_neuron, Flux.relu),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.identity),
    # Add normalization layer
    Flux.BatchNorm(n_neuron, Flux.relu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.identity),
    # Add normalization layer
    Flux.BatchNorm(n_neuron, Flux.relu),
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
        # Add normalization layer
        Flux.BatchNorm(n_neuron, Flux.relu),
        # Second Layer
        Flux.Dense(n_neuron => n_neuron, Flux.identity),
        # Add normalization layer
        Flux.BatchNorm(n_neuron, Flux.relu),
        # Third layer
        Flux.Dense(n_neuron => n_neuron, Flux.identity),
        # Add normalization layer
        Flux.BatchNorm(n_neuron, Flux.relu),
        # Output layer
        Flux.Dense(n_neuron => n_env, Flux.identity)
    )
)

## =============================================================================

println("Define MetricChain...")

# Define mlp chain
mlp_chain = Flux.Chain(
    # First layer
    Flux.Dense(n_env => n_neuron, Flux.relu),
    # Second layer
    Flux.Dense(n_neuron => n_neuron, Flux.relu),
    # Third layer
    Flux.Dense(n_neuron => n_neuron, Flux.relu),
)

# Define layers for the diagonal and lower triangular part of the covariance
# matrix
diag = Flux.Dense(n_neuron => n_latent, Flux.identity)
lower = Flux.Dense(
    n_neuron => n_latent * (n_latent - 1) ÷ 2, Flux.identity
)

# Build metric chain
metric_chain = AutoEncode.RHVAEs.MetricChain(mlp_chain, diag, lower)

## =============================================================================

# Initialize rhvae
rhvae = AutoEncode.RHVAEs.RHVAE(
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
    "./output/model.jld2",
    Dict("model" => rhvae, "model_state" => Flux.state(rhvae))
)