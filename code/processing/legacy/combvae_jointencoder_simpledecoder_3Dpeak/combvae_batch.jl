## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import package to revise module
import Revise
# Import project package
import Antibiotic

# Import library to list files
import Glob

# Import libraries to work with DataFrames
import DataFrames as DF
import CSV

# Import ML libraries
import Flux
import AutoEncode

# Import library to compute nearest neighbors
import NearestNeighbors

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase

Random.seed!(42)
##

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 10_000
# Define number of epochs
n_epoch = 50_000
# Define how often to compute error
n_error = 1000

# Define number of primary samples
n_primary = 8
# Define number of secondary samples
n_secondary = 8
# Define number of neighbors to consider
k_neighbors = 20

# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 20
# Define dimensionality of latent space
n_latent = 2

# Define parameter scheduler
epoch_change = [1]
learning_rates = [10^-3]
# Generate dictionary to pudate learning rate
learning_epoch = Dict(zip(epoch_change, learning_rates))

# Defie loss function arguments
β = 0.1f0

##

println("Generating synthetic data...")

# Define function
f(x₁, x₂) = 10 * exp(-(x₁^2 + x₂^2))

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

# Compute nearest neighbor tree
nn_tree = NearestNeighbors.BruteTree(data_std)

##

println("Setting output directories...")
# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

# Define output filename
fname = "./output/$(n_epoch)_epoch/combvae_$(n_latent)dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("$(n_latent)-D was already processed")
end # if

##

# Define VAE architecture

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

##

println("Initializing VAE...")

# Initialize encoder
encoder = AutoEncode.VAEs.JointEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Initialize decoder
decoder = AutoEncode.VAEs.SimpleDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Set VAE
vae = AutoEncode.VAEs.VAE(encoder, decoder)

##

println("Writing down metadata to README.md file")

# Define text to go into README
readme = "
# `$(@__FILE__)`
## VAE type
`typeof = $(typeof(vae))`
## Latent space dimensionalities to train
`n_latent = $(n_latent)`
## Number of epochs
`n_epoch = $(n_epoch)`
## Number of epochs between error evaluations
`n_error = $(n_error)`
## Number of primary samples
`n_primary = $(n_primary)`
## Number of secondary samples
`n_secondary = $(n_secondary)`
## Number of neighbors to consider
`k_neighbors = $(k_neighbors)`
## Number of hidden non-linear layers
`n_hidden = $(n_hidden)`
## Number of neurons per layer
`n_neuron = $(n_neuron)`
# Define parameter scheduler
`epoch_change = $(epoch_change)`
`learning_rates = $(learning_rates)`
"

# Write README file into memory
open("./output/$(n_epoch)_epoch/README.md", "w") do file
    write(file, readme)
end

##

# Define array where to store MSE, loss, and KL divergence
vae_mse = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
vae_loss = similar(vae_mse)
vae_kl = similar(vae_mse)

# Compute mse
vae_mse[1] = StatsBase.mean(
    Flux.mse.(eachcol(data_std), eachcol(vae(data_std)))
)
# Compute mean loss
vae_loss[1] = StatsBase.mean(
    AutoEncode.VAEs.loss.(Ref(vae), eachcol(data_std); β=β)
)

# Compute mean KL divergence
vae_kl[1] = StatsBase.mean([
    AutoEncode.VAEs.loss_terms(vae, col, logpost=false)[2]
    for col in eachcol(data_std)
])

println("MSE: $(vae_mse[1])")
println("loss: $(vae_loss[1])")

##

println("Training VAE...")
# Initialize error counter
global error_counter = 2

# Define initial learning rate
η = learning_epoch[1]
# Define opt state
opt_state = Flux.Train.setup(Flux.Adam(η), vae)

# Loop through epochs
for j = 1:n_epoch
    # Select batch
    data_batch = AutoEncode.utils.locality_sampler(
        data_std, nn_tree, n_primary, n_secondary, k_neighbors
    )

    # Train network
    AutoEncode.VAEs.train!(
        vae, data_batch, opt_state; average=true, loss_kwargs=Dict(:β => β)
    )

    # Check if learning rate changed
    if j ∈ epoch_change
        # Update optimizer with new learning rate
        Flux.adjust!(opt_state, learning_epoch[j])
    end # if

    # Check if error should be computed
    if j % n_error == 0
        println("Epoch # $j / $(n_epoch)")

        # Compute mean square error
        vae_mse[error_counter] = StatsBase.mean(
            Flux.mse.(eachcol(data_std), eachcol(vae(data_std)))
        )
        # Compute mean loss
        vae_loss[error_counter] = StatsBase.mean(
            AutoEncode.VAEs.loss.(Ref(vae), eachcol(data_std); β=β)
        )
        # Compute mean KL divergence
        vae_kl[error_counter] = StatsBase.mean([
            AutoEncode.VAEs.loss_terms(vae, col, logpost=false)[2]
            for col in eachcol(data_std)
        ])

        println("MSE: $(vae_mse[error_counter])")
        println("loss: $(vae_loss[error_counter])")
        # Update counter
        global error_counter += 1
    end # if
end # for

##

# Save output
println("Saving results...")

BSON.bson(
    fname,
    vae=vae,
    mse=vae_mse,
    loss=vae_loss,
    kl=vae_kl,
    data=data,
    data_std=data_std,
)

println("Done!")
