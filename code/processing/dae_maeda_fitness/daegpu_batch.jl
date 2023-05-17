## 
println("Loading packages...")
# Load project package
@load_pkg Antibiotic

# Import project package
import Antibiotic

# Import AutoEncode package
import AutoEncode

# Import libraries to work with DataFrames
import DataFrames as DF
import CSV

# Import ML libraries
import Flux
import NNlib
import CUDA

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase
import Distributions

Random.seed!(42)
##

# # Check if GPU is available
# if !(CUDA.functional())
#     error("GPU is not available")
# end # if

# # Check CUDA information
# CUDA.versioninfo()

##

# Define number of epochs
n_epoch = 100_000
# Define how often to compute error
n_error = 1_000
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 100
# Define dimensionality of latent space
latent_dim = 3
# Define noise probability
noise_prob = 0.2
# Define parameter scheduler
epoch_change = [1, 10^4, 10^5, 5 * 10^5, 10^6]
learning_rates = [10^-5, 10^-6, 10^-6.5, 10^-7, 10^-8]

##

println("Setting output directories...\n")

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

# Define output filename
fname = "./output/$(n_epoch)_epoch/dae_$(latent_dim)dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("$(latent_dim)-D was already processed")
end # if

##

##

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
ic50_std = StatsBase.transform(dt, ic50_mat) |> Flux.gpu

##

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(latent_dim)`
## Number of epochs
`n_epoch = $(n_epoch)`
## Number of epochs between error evaluations
`n_error = $(n_error)`
## Batch size
`n_batch = $(n_batch)`
## Number of hidden non-linear layers
`n_hidden = $(n_hidden)`
## Number of neurons per layer
`n_neuron = $(n_neuron)`
## Noise probability
`noise_prob = $(noise_prob)`
## learning rate scheduler
`epoch_change = $(epoch_change)`
`learning_rates = $(learning_rates)`
`AutoEncode.utils.step_scheduler(x, epoch_change, learning_rates)`
"""

# Write README file into memory
open("./output/$(n_epoch)_epoch/README.md", "w") do file
    write(file, readme)
end

##

println("Definining encoder architecture...\n")

#

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.softplus], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.softplus], n_hidden)

##

# Initialize full Flux autoencoder to take advantage of the AEs module. Note: we
# will only train the encoder section since the output should be the learned
# latent space.
ae = AutoEncode.AEs.ae_init(
    n_env,
    latent_dim,
    latent_activation,
    output_activation,
    encoder,
    encoder_activation,
    decoder,
    decoder_activation,
) |> Flux.gpu

##

println("Setting output directories...")
# Define output filename
fname = "./output/$(n_epoch)_epoch/dae_$(latent_dim)dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("$(latent_dim)-D was already processed")
end # if

##

println("Definining loss function as MSE...")
# Define loss function. The loss function is the MSE between the encoder output
# and the learned latent space built from the fitness data only
loss(x̂, x) = Flux.mse(AutoEncode.recon(ae, x̂), x)

##

println("Training encoder...")

# Define array where to store errors on training data
ae_mse = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
# Evaluate initial error
ae_mse[1] = loss(ic50_std, ic50_std) |> Flux.cpu

# Initialize error counter
global mse_counter = 2

# Loop through epochs
for j = 1:n_epoch
    # Sample data indexes to be used in this epoch
    batch_idx = StatsBase.sample(
        1:n_samples, n_batch, replace=false
    )
    # Extract minibatch ic50 data
    ic50_batch = @view ic50_std[:, batch_idx]
    # Generate noisy data
    ic50_noise = Random.rand(
        Distributions.Bernoulli(1 - noise_prob), size(ic50_batch)...
    ) .* ic50_batch

    # Train network
    Flux.train!(
        loss,
        Flux.params(ae.encoder, ae.decoder),
        zip(eachcol(ic50_noise), eachcol(ic50_batch)),
        Flux.ADAM(
            AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
        )
    )

    # Print progress report
    if (j % n_error == 0)
        println("Epoch # $j")
        ae_mse[mse_counter] = loss(ic50_batch, ic50_batch) |> Flux.cpu

        println("MSE: $(ae_mse[mse_counter])")

        # Update counter
        global mse_counter += 1
    end # if
end # for

println("Saving data into memory...\n")

BSON.bson(
    fname,
    ae=Flux.cpu(ae),
    mse=ae_mse,
    data=ic50_mat,
    data_std=Flux.cpu(ic50_std),
)

println("Done!")
