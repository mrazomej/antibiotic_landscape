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

# Define InfoMaxVAE parameters
α = 10.0f0
β = 1.0f0

# Define number of epochs
n_epoch = 100_000
# Define how often to compute error
n_error = 1_000
# Define batch size
n_batch = 64
# Define number of hidden layers
n_hidden = 4
# Define number of neurons in non-linear hidden layers
n_neuron = 20
# Define dimensionality of latent space
latent_dim = 2
# Define parameter scheduler
epoch_change = [1, 10^4, 3 * 10^4, 10^5]
learning_rates = [10^-6, 10^-6, 10^-6, 10^-6]

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

println("Definining InfoMaxVAE architecture...\n")

#

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.relu], n_hidden)

# Define MLP layers and activation functions
mlp = repeat([n_neuron], n_hidden)
mlp_activation = repeat([Flux.relu], n_hidden)

# Define output activation
mlp_output_activation = Flux.identity

##

println("Training InfoMAxVAE...\n")
##

println("Setting file name...")
# Define output filename
fname = "./output/$(n_epoch)_epoch/" *
        "infomaxvae_$(lpad(latent_dim, 2, "0"))dimensions.bson"

# Check if autoencoder has already been trained
if isfile(fname)
    error("InfoMaxVAE with $(latent_dim)D latent space was already trained")
end # if

##

# Initialize full Flux autoencoder to take advantage of the AEs module.
vae = AutoEncode.InfoMaxVAEs.infomaxvae_init(
    n_env,
    latent_dim,
    latent_activation,
    output_activation,
    encoder,
    encoder_activation,
    decoder,
    decoder_activation,
    mlp,
    mlp_activation,
    mlp_output_activation,
)

##

println("Training InfoMaxVAE...")

# Define array where to store loss function
vae_loss = Float32[]

# Define array where to store mutual information
vae_mi = Float32[]

# Define array where to store errors on training data
vae_mse = Float32[]


# Evaluate loss function
push!(
    vae_loss,
    AutoEncode.InfoMaxVAEs.loss(
        vae.vae,
        vae.mlp,
        ic50_std,
        ic50_std[:, Random.shuffle(1:end)];
        α=α,
        β=β
    )
)

# Evaluate mutual information
push!(vae_mi, AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, ic50_std))

# Evaluate initial error
push!(vae_mse, Flux.cpu(Flux.mse(vae.vae(ic50_std), ic50_std)))

# Define optimizer to be used for the VAE
vae_opt = Flux.Train.setup(
    Flux.ADAM(
        AutoEncode.utils.step_scheduler(1, epoch_change, learning_rates)
    ),
    vae.vae
)

# Define optimizer to be used for the MLP
mlp_opt = Flux.Train.setup(
    Flux.ADAM(
        AutoEncode.utils.step_scheduler(1, epoch_change, learning_rates)
    ),
    vae.mlp
)

# Loop through epochs
for j = 1:n_epoch
    # Update optimizer learning rate
    Flux.Optimisers.adjust!(
        vae_opt,
        AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
    )
    # Update optimizer learning rate
    Flux.Optimisers.adjust!(
        mlp_opt,
        AutoEncode.utils.step_scheduler(j, epoch_change, learning_rates)
    )

    # Sample data indexes to be used in this epoch
    batch_idx = StatsBase.sample(1:n_samples, n_batch, replace=false)
    ic50_batch = @view ic50_std[:, batch_idx]

    # Train network
    AutoEncode.InfoMaxVAEs.train!(
        vae,
        ic50_batch,
        vae_opt,
        mlp_opt;
        loss_kwargs=(α=α, β=β)
    )

    # Print progress report
    if (j % n_error == 0)
        println("Epoch #$j\n")
        # Evaluate loss function
        push!(
            vae_loss,
            Flux.cpu(
                AutoEncode.InfoMaxVAEs.loss(
                    vae.vae,
                    vae.mlp,
                    ic50_std,
                    ic50_std[:, Random.shuffle(1:end)]
                )
            )
        )

        # Evaluate mutual information
        push!(
            vae_mi,
            AutoEncode.InfoMaxVAEs.mutual_info_mlp(vae, ic50_std)
        )

        # Evaluate error
        push!(vae_mse, Flux.cpu(Flux.mse(vae.vae(ic50_std), ic50_std)))

        logP, kl_div, mi = Flux.cpu(
            AutoEncode.InfoMaxVAEs.loss_terms(
                vae.vae,
                vae.mlp,
                ic50_std,
                ic50_std[:, Random.shuffle(1:end)]
            )
        )

        println("⟨log P(x|z)⟩ = $(logP)")
        println("Dₖₗ(qᵩ(x | z) || P(x)) = $(kl_div)")
        println("I(x;z) = $(mi)")
        # println("Loss: $(vae_loss[end])")

        # println("Mutual Info: $(vae_mi[end])")

        # println("MSE: $(vae_mse[end])\n")
    end # if
end # for

println("Saving data into memory...\n")

BSON.bson(
    fname,
    ae=Flux.cpu(vae),
    mse=vae_mse,
    mi=vae_mi,
    loss=vae_loss,
    ic50_std=Flux.cpu(ic50_std),
)

println("Done!")