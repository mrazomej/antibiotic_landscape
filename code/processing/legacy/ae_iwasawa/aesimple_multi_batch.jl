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
import NNlib
import SimpleChains

# Import library to save models
import BSON

# Import basic math
import Random
import LinearAlgebra
import StatsBase

Random.seed!(42)
##

println("Reading metadata...")

# Define number of epochs
n_epoch = 1_000_000
# Define how often to compute error
n_error = 10_000
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 20
# Define range of latent space to train on
latent_dims = collect(1:8)
# Define parameter scheduler
epoch_change = [1, 10^4, 10^5, 5 * 10^5, 10^6]
learning_rates = [10^-3, 10^-3, 10^-3.5, 10^-4, 10^-4.5];

##

# Define positions for each of the ranges to evaluate
steps = collect(1:n_error:n_epoch)

# List ranges for each iteration
ranges = [steps[i]:steps[i+1]-1 for i = 1:length(steps)-1]
# If needed, add last step
if maximum(ranges[end]) < n_epoch
    push!(ranges, maximum(ranges[end])+1:n_epoch)
end # if

# Define learning rates
η_array = [
    Antibiotic.ml.step_scheduler(x, epoch_change, learning_rates)
    for x in minimum.(ranges)
]

##

##

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/data/Iwasawa_2022"

# Load file into memory
df_ic50 = CSV.read("$(data_dir)/iwasawa_ic50_tidy.csv", DF.DataFrame)

# Locate strains with missing values
missing_strains = unique(df_ic50[ismissing.(df_ic50.log2ic50), :strain])

# Remove data
df_ic50 = df_ic50[[x ∉ missing_strains for x in df_ic50.strain], :]


# Group data by strain and day
df_group = DF.groupby(df_ic50, [:strain, :day])

# Extract unique drugs to make sure the matrix is built correctly
drug = sort(unique(df_ic50.drug))

# Initialize matrix to save ic50 values
ic50_mat = Matrix{Float32}(undef, length(drug), length(df_group))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Check that the stress are in the correct order
    if all(data.drug .== drug)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.log2ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat);

##

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir("./output/$(n_epoch)_epoch")
    mkdir("./output/$(n_epoch)_epoch")
end # if

##

println("Writing down metadata to README.md file")

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Latent space dimensionalities to train
`latent_dim = $(latent_dims)`
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
## learning rate scheduler
`epoch_change = $(epoch_change)`
learning_rates = $(learning_rates)`
`Antibiotic.ml.step_scheduler(x, epoch_change, learning_rates)`
"""

# Write README file into memory
open("./output/$(n_epoch)_epoch/README.md", "w") do file
    write(file, readme)
end

##

# Define autoencoder architecture

# Define latent space activation function
latent_activation_sc = SimpleChains.identity
latent_activation_flux = Flux.identity
# Define output layer activation function
output_activation_sc = SimpleChains.identity
output_activation_flux = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation_sc = repeat([SimpleChains.relu], n_hidden)
encoder_activation_flux = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation_sc = repeat([SimpleChains.relu], n_hidden)
decoder_activation_flux = repeat([Flux.relu], n_hidden)

##

println("Initializing autoencoders with different dimensionalities...")

# Initialize autoencoders for each dimensionality
ae_sc_template = [
    Antibiotic.ml.AEs.SimpleAutoencoder(
        n_env,
        l,
        latent_activation_sc,
        output_activation_sc,
        encoder,
        encoder_activation_sc,
        decoder,
        decoder_activation_sc,
    ) for l in latent_dims
]
ae_flux_template = [
    Antibiotic.ml.AEs.Autoencoder(
        n_env,
        l,
        latent_activation_flux,
        output_activation_flux,
        encoder,
        encoder_activation_flux,
        decoder,
        decoder_activation_flux,
    ) for l in latent_dims
]

# Initialize parameters
param_list = [SimpleChains.init_params(ae) for ae in ae_sc_template]

# Initialize autoencoders with error function
ae_loss_list = [
    SimpleChains.add_loss(ae, SimpleChains.SquaredLoss(ic50_std))
    for ae in ae_sc_template
]

##

println("Initializing Autoenconder(s) training")
# Loop 
Threads.@threads for i = 1:length(ae_loss_list)
    println("Training $(latent_dims[i])D autoencoder on Kinsler data")

    # Define output filename
    fname = "./output/$(n_epoch)_epoch/ae_$(latent_dims[i])dimensions.bson"

    # Check if autoencoder has already been trained
    if isfile(fname)
        println("$(latent_dims[i])-D was already processed")
        continue
    end # if

    # Define array where to store errors on training data
    ae_mse_sc = Vector{Float32}(undef, n_epoch ÷ n_error + 1)
    ae_mse_flux = similar(ae_mse_sc)

    # Evaluate initial error
    ae_mse_sc[1] = ae_loss_list[i](ic50_std, param_list[i])
    ae_mse_flux[1] = Flux.mse(
        Antibiotic.ml.AEs.simple_to_flux(
            collect(param_list[i]), ae_flux_template[i]
        )(ic50_std),
        ic50_std
    )

    # Loop through training ranges
    for (j, r) in enumerate(ranges)

        println("Epochs $(r) for $(latent_dims[i]) AE")
        # Train autoencoder
        SimpleChains.train_batched!(
            similar(param_list[i]),
            param_list[i],
            ae_loss_list[i],
            ic50_std,
            SimpleChains.ADAM(η_array[j]),
            length(r);
            batchsize=n_batch
        )

        # Evaluate SimpleChains error
        ae_mse_sc[j+1] = ae_loss_list[i](ic50_std, param_list[i])

        # Evaluate Flux MSE
        ae_mse_flux[j+1] = Flux.mse(
            Antibiotic.ml.AEs.simple_to_flux(
                collect(param_list[i]), ae_flux_template[i]
            )(ic50_std),
            ic50_std
        )
    end # for
    println("Converting $(fname) output from SimpleChains to Flux")
    BSON.bson(
        fname,
        ae=Antibiotic.ml.AEs.simple_to_flux(
            collect(param_list[i]), ae_flux_template[i]
        ),
        mse_sc=ae_mse_sc,
        mse_flux=ae_mse_flux,
        data=ic50_mat,
        data_std=ic50_std,
    )
end # for

println("Done!")
