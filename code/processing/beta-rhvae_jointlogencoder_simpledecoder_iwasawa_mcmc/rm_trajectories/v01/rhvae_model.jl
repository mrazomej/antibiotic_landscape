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
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Load standardized mean data
logic50_mean_std = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_mean_std"]

# Load metadata
df_meta = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_meta"]

# Define number of environmenst
n_env = size(logic50_mean_std, 1)

## =============================================================================

println("Excluding a few lineages from training...")

# Group data by :strain, :evolution
df_group = DF.groupby(df_meta, [:strain, :evolution])

# Obtain number of biological replicates per group
rep_num = [length(unique(d.design)) for d in df_group]

# Count how many times each rep_num appears
rep_count = StatsBase.countmap(rep_num)

# Initialize list of strain_num to exclude from training
strain_num_exclude = Int[]

# Loop through each group
for (i, data) in enumerate(df_group)
    # If number of replicates is > 1, add first strain_num to list
    if rep_num[i] > 1
        push!(strain_num_exclude, first(unique(data.strain_num)))
    end
end # for

# Locate indexes to exclude and to keep
idx_keep = .!in.(df_meta.strain_num, Ref(strain_num_exclude))
idx_rm = .!idx_keep

# Index data to keep and remove
logic50_mean_std_keep = logic50_mean_std[:, idx_keep]
logic50_mean_std_rm = logic50_mean_std[:, idx_rm]
df_meta_keep = df_meta[idx_keep, :]
df_meta_rm = df_meta[idx_rm, :]

## =============================================================================

# Locate current directory
path_dir = pwd()

## =============================================================================

# Select centroids via k-means
centroids_data = AutoEncoderToolkit.utils.centroids_kmedoids(
    logic50_mean_std_keep, n_centroids
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
    Dict(
        "model" => rhvae,
        "model_state" => Flux.state(rhvae),
        "idx_keep" => idx_keep,
        "idx_rm" => idx_rm,
    )
)
