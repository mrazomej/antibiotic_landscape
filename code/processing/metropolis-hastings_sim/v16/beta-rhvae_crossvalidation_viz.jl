## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DimensionalData as DD
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
using PDFmerger: append_pdf!
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
original_state_dir = "$(vae_dir)/model_state"
# Define cross-validation directory
cross_state_dir = "$(vae_dir)/model_crossvalidation_state"
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/vae"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading model...")

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = (K=K, ϵ=ϵ, βₒ=βₒ,)

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
original_model_states = Glob.glob("$(original_state_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over files
for f in original_model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", f).captures[1])
    # Load model_state file
    f_load = JLD2.load(f)
    # Extract values
    loss_train = f_load["loss_train"]
    loss_val = f_load["loss_val"]
    mse_train = f_load["mse_train"]
    mse_val = f_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => model_file,
        :model_state => f,
        :model_type => "original",
        :split => "train",
        :train_frac => 0.85,
        :val_frac => 0.15,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for f in model_states

## =============================================================================

# List cross-validation split directories
cross_model_splits = readdir(cross_state_dir)

for split in cross_model_splits
    # List model states
    cross_model_states = Glob.glob(
        "$(cross_state_dir)/$(split)/*.jld2"[2:end], "/"
    )
    # Extract train and validation fractions from split name
    train_frac = parse(Float64, match(r"(\d+)train", split).captures[1])
    val_frac = parse(Float64, match(r"(\d+)val", split).captures[1])
    # Loop over files
    for f in cross_model_states
        # Extract epoch number from file name using regular expression
        epoch = parse(Int, match(r"epoch(\d+)", f).captures[1])
        # Load model_state file
        f_load = JLD2.load(f)
        # Extract values
        loss_train = f_load["loss_train"]
        loss_val = f_load["loss_val"]
        mse_train = f_load["mse_train"]
        mse_val = f_load["mse_val"]
        # Generate temporary dataframe to store metadata
        df_tmp = DF.DataFrame(
            :epoch => epoch,
            :loss_train => loss_train,
            :loss_val => loss_val,
            :mse_train => mse_train,
            :mse_val => mse_val,
            :model_file => model_file,
            :model_state => f,
            :model_type => "cross",
            :split => replace(split, "split_" => ""),
            :train_frac => train_frac / 100,
            :val_frac => val_frac / 100,
        )
        # Append temporary dataframe to main dataframe
        global df_meta = DF.vcat(df_meta, df_tmp)
    end # for f in model_states
end # for split in cross_model_splits

## =============================================================================

println("Plotting training loss...")

# Group by model type
df_group = DF.groupby(df_meta, [:model_type, :split])

# Initialize figure
fig = Figure(size=(900, 600))

# Define number of rows and columns
n_row = 2
n_col = 3

# Loop over groups
for (i, df) in enumerate(df_group)
    # Obtain row and column indices
    row = (i - 1) ÷ n_col + 1
    col = (i - 1) % n_col + 1
    # Initialize axis
    ax = Axis(
        fig[row, col],
        xlabel="epoch",
        ylabel="mean squared error",
        title="$(first(df.model_type)) $(replace(first(df.split), "_" => ":"))",
    )
    # Define colors
    if first(df.model_type) == "original"
        colors = ColorSchemes.Paired_12[1:2]
    else
        colors = ColorSchemes.Paired_12[3:4]
    end
    # Plot training loss
    lines!(
        ax,
        df.epoch,
        df.mse_train,
        color=colors[1],
        label="training",
    )
    # Plot validation loss
    lines!(
        ax,
        df.epoch,
        df.mse_val,
        color=colors[2],
        label="validation",
    )
    # Add legend
    axislegend(ax, position=:rt)
end # for (i, df) in enumerate(df_group)

save("$(fig_dir)/rhvae_loss_cross.pdf", fig)
save("$(fig_dir)/rhvae_loss_cross.png", fig)

fig

## =============================================================================

# Group by split
df_group = DF.groupby(df_meta[df_meta.model_type.=="cross", :], :split)

# Initialize figure
fig = Figure(size=(500, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="fraction of data used for training",
    ylabel="mean squared error",
)

# Loop over groups
for (i, df) in enumerate(df_group)
    # Plot training loss for last epoch
    scatter!(
        ax,
        first(df.train_frac),
        last(DF.sort(df, :epoch).mse_train),
        color=ColorSchemes.Paired_12[3],
        label="cross training",
        markersize=12,
    )
    # Plot validation loss for last epoch
    scatter!(
        ax,
        first(df.train_frac),
        last(DF.sort(df, :epoch).mse_val),
        color=ColorSchemes.Paired_12[4],
        label="cross validation",
        markersize=12,
    )
end # for (i, df) in enumerate(df_group)

# Extract original mse
df_original = df_meta[df_meta.model_type.=="original", :]

# Add original mse as horizontal line
hlines!(
    ax,
    last(DF.sort(df_original, :epoch).mse_train),
    color=ColorSchemes.Paired_12[1],
    linestyle=:dash,
    label="original training (0.85)",
    linewidth=2,
)
hlines!(
    ax,
    last(DF.sort(df_original, :epoch).mse_val),
    color=ColorSchemes.Paired_12[2],
    linestyle=:dash,
    label="original validation (0.15)",
    linewidth=2,
)
# Add legend
axislegend(ax, position=:rt, unique=true, merge=true)

# Save figure
save("$(fig_dir)/rhvae_loss_cross_trainfrac.pdf", fig)
save("$(fig_dir)/rhvae_loss_cross_trainfrac.png", fig)

fig

## =============================================================================

println("Loading data into memory...")

# Define the subsampling interval
n_sub = 10

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))
# Define number of training environments
n_env_train = n_env ÷ 2

# Subsample time series and environments
fitnotype_profiles = fitnotype_profiles[
    time=DD.At(t_init:n_sub:t_final),
]

# Extract fitness data bringing the fitness dimension to the first dimension
fit_data = permutedims(fitnotype_profiles.fitness.data, (5, 1, 2, 3, 4, 6))
# Reshape the array to a Matrix
fit_data = reshape(fit_data, size(fit_data, 1), :)

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        log.(fitnotype_profiles.fitness.data),
        dims=[5]),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Loading models...")

# Load model
rhvae_original = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(
    rhvae_original,
    JLD2.load(
        df_meta[df_meta.model_type.=="original", :model_state][end]
    )["model_state"]
)
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae_original)

# Load model
rhvae_cross = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(
    rhvae_cross,
    JLD2.load(
        df_meta[df_meta.model_type.=="cross", :model_state][end]
    )["model_state"]
)
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae_cross)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Extract "training" data
log_fitnotype_std_train = log_fitnotype_std[
    landscape=DD.At(1:n_env_train),
    evo=DD.At(1:n_env_train),
]

# Map data to latent space
dd_latent_original = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae_original.vae.encoder(slice).μ,
            log_fitnotype_std_train.data,
            dims=[5]);
        dims=1
    ),
    (
        log_fitnotype_std_train.dims[2:4]...,
        latent,
        log_fitnotype_std_train.dims[6]
    ),
)

# Map data to latent space
dd_latent_cross = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae_cross.vae.encoder(slice).μ,
            log_fitnotype_std_train.data,
            dims=[5]);
        dims=1
    ),
    (
        log_fitnotype_std_train.dims[2:4]...,
        latent,
        log_fitnotype_std_train.dims[6]
    ),
)

# Stack latent spaces
dd_latent = DD.DimStack(
    (original=dd_latent_original, cross=dd_latent_cross),
)

## =============================================================================

println("Plotting latent space coordinates...")

# Initialize figure
fig = Figure(size=(600, 325))

# Add axis
ax_original = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="Original RHVAE\n (envs 1-$(n_env_train))",
    aspect=AxisAspect(1)
)
# Add axis
ax_cross = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="Cross-validated RHVAE\n (envs $(n_env_train+1)-$(n_env))",
    aspect=AxisAspect(1)
)

# Plot latent space
scatter!(
    ax_original,
    vec(dd_latent.original[latent=DD.At(:latent1)]),
    vec(dd_latent.original[latent=DD.At(:latent2)]),
    markersize=5,
    color=ColorSchemes.seaborn_colorblind[1],
)
# Plot latent space
scatter!(
    ax_cross,
    vec(dd_latent.cross[latent=DD.At(:latent1)]),
    vec(dd_latent.cross[latent=DD.At(:latent2)]),
    markersize=5,
    color=ColorSchemes.seaborn_colorblind[2],
)

save("$(fig_dir)/rhvae_latent_space_cross.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_cross.png", fig)

fig