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

# Define boolean wether it should be evaluated with gpu
gpu_eval = false

# Define emphasis values
α_mask = 1.0f0
β_retain = 0.0f0

# Define number of cross-validation events
n_cross = 10
# Define fraction of data for training
train_frac = 0.75
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
latent_dim = 8
# Define noise probability
noise_prob = 0.25
# Define parameter scheduler
epoch_change = [1, 10^4, Int64(7.5 * 10^4), 10^5]
learning_rates = [10^-4, 10^-4, 10^-4.25, 10^-4.5]

##
# Check if GPU is available
if gpu_eval & !(CUDA.functional())
    error("GPU is not available")
end # if

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
## Number of cross-validation events
`n_cross = $(n_cross)`
## Fraction of data used for training
`train_frac = $(train_frac)`
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
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

##

# Generate splits for reproducibility
train_idx = [
    StatsBase.sample(
        1:n_samples, Int64(round(n_samples * train_frac)), replace=false
    )
    for i = 1:n_cross
]

##

println("Definining loss function as MSE...")
# Define loss function. The loss function is the MSE between the encoder output
# and the learned latent space built from the fitness data only. Notice that
# this function can take a mini-batch matrix as input more efficiently since it
# runs the inputs through the autoencoder all together, which is much more
# efficient than doing it individually.
function loss(model, data)
    # Unpack data
    x̂, x = data
    # Reconstruct input
    x̂_pred = model(x̂)
    # Evaluate loss function
    return sum(Flux.mse.(eachcol(x̂_pred), eachcol(x))) / size(x, 2)
end # function

# Define emphasis loss function
function emph_loss(model, data)
    # unpack data
    x̂, x, emph_mask = data
    # Reconstruct input
    x̂_mask = model(x̂) .* emph_mask
    x_mask = x .* emph_mask
    # Evaluate loss function
    return sum(Flux.mse.(eachcol(x̂_mask), eachcol(x_mask))) / size(x, 2)
end # function

##
# Loop through cross validation
for cross in 1:n_cross
    println("Training $(cross) iteration...\n")
    ##

    println("Setting output directories...")
    # Define output filename
    fname = "./output/$(n_epoch)_epoch/" *
            "edae_$(lpad(cross, 2, "0"))_$(lpad(latent_dim, 2, "0"))dimensions.bson"

    # Check if autoencoder has already been trained
    if isfile(fname)
        println("Iteration $(cross) for $(latent_dim)D AE was already processed")
        continue
    end # if

    ##

    # Extract train and test data
    data_train = ic50_std[:, train_idx[cross]]
    data_test = ic50_std[:, setdiff(1:n_samples, train_idx[cross])]

    # Initialize full Flux autoencoder to take advantage of the AEs module.
    ae = AutoEncode.AEs.ae_init(
        n_env,
        latent_dim,
        latent_activation,
        output_activation,
        encoder,
        encoder_activation,
        decoder,
        decoder_activation,
    )

    # Check if model should be run on gpu
    if gpu_eval
        # Load autoencoder parameters to GPU
        ae.encoder = Flux.fmap(CUDA.cu, ae.encoder)
        ae.decoder = Flux.fmap(CUDA.cu, ae.decoder)
    end # if

    ##

    println("Generating random noise")
    rnd_mask = Random.rand(
        Distributions.Bernoulli(1 - noise_prob), n_env, n_epoch
    )

    # Initialize array to store mask for all data
    rnd_noise = Array{Bool}(undef, n_env, n_batch, n_epoch)
    # Loop through epochs
    for i = 1:n_epoch
        rnd_noise[:, :, i] = hcat(repeat([rnd_mask[:, i]], n_batch)...)
    end # for

    # Check if GPU should be used
    if gpu_eval
        rnd_noise = Flux.gpu(rnd_noise)
    end # if

    ##
    println("Generating emphasis mask")
    # Initialize emphasis mask
    emph_mask = Array{Float32}(undef, size(rnd_noise)...)

    # Add emphasis values
    emph_mask[.!rnd_noise] .= √(α_mask)
    emph_mask[rnd_noise] .= √(β_retain)

    # Check if GPU should be used
    if gpu_eval
        emph_mask = Flux.gpu(emph_mask)
    end # if

    ##

    println("Training encoder...")

    # Define array where to store errors on training data
    ae_mse_train = Float32[]
    # Define array where to store errors on test data
    ae_mse_test = Float32[]

    # Evaluate initial error
    push!(ae_mse_train, Flux.cpu(loss(ae, (data_train, data_train,))))
    push!(ae_mse_test, Flux.cpu(loss(ae, (data_test, data_test,))))

    # Define optimizer to be used 
    ae_opt = Flux.Train.setup(
        Flux.ADAM(
            AutoEncode.utils.step_scheduler(1, epoch_change, learning_rates)
        ),
        ae
    )
    # Loop through epochs
    for j = 1:n_epoch
        # Sample data indexes to be used in this epoch
        batch_idx = StatsBase.sample(
            1:Int64(round(n_samples * train_frac)), n_batch, replace=false
        )
        # Extract minibatch ic50 data
        ic50_batch = data_train[:, batch_idx]
        # Generate noisy data
        ic50_noise = rnd_noise[:, :, j] .* ic50_batch

        # Train network
        Flux.train!(
            emph_loss,
            ae,
            [((ic50_noise, ic50_batch, emph_mask[:, :, j]),),],
            ae_opt
        )

        # Update optimizer learning rate
        Flux.Optimisers.adjust!(
            ae_opt,
            AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
        )

        # Print progress report
        if (j % n_error == 0)
            println("Epoch # $j")
            push!(ae_mse_train, Flux.cpu(loss(ae, (data_train, data_train))))
            push!(ae_mse_test, Flux.cpu(loss(ae, (data_test, data_test))))

            println("MSE train: $(ae_mse_train[end])")
            println("MSE test: $(ae_mse_test[end])")
        end # if
    end # for

    println("Saving data into memory...\n")

    BSON.bson(
        fname,
        ae=Flux.cpu(ae),
        mse_train=ae_mse_train,
        mse_test=ae_mse_test,
        data_train=Flux.cpu(data_train),
        data_test=Flux.cpu(data_test),
    )

    println("Done with iteration $(cross)!")
end # for


println("Done!")