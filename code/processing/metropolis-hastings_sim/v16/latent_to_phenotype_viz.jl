## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DimensionalData as DD

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
state_dir = "$(vae_dir)/model_state"
# Define directory for neural network
mlp_dir = "$(git_root())/output$(out_prefix)/mlp"
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/mlp"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation results...")

# Define the subsampling interval
n_sub = 10

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))

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

println("Loading model...")

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Mapping data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

# Reorder dimensions
dd_latent = permutedims(dd_latent, (4, 1, 2, 3, 5))

## =============================================================================

println("Loading loss vectors...")
# List neural network files
mlp_files = sort(Glob.glob("$(mlp_dir)/*split*.jld2"[2:end], "/"))

# Initialize dictionary to store loss vectors
loss_train_dict = Dict{Float32,Vector{Float32}}()
loss_val_dict = Dict{Float32,Vector{Float32}}()
# Loop through neural network files
for mlp_file in mlp_files
    # Extract split fraction from file name
    split_frac = parse(
        Float32,
        match(r"split(.*).jld2", mlp_file).captures[1]
    )
    # Load loss vectors
    loss_train_dict[split_frac] = JLD2.load(mlp_file)["loss_train"]
    loss_val_dict[split_frac] = JLD2.load(mlp_file)["loss_val"]
end # end loop through neural network files

## =============================================================================

println("Plotting loss vectors...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis for training loss
ax_train = Axis(
    fig[1, 1],
    title="Training",
    xlabel="epoch",
    ylabel="loss",
    yscale=log10,
)
# Add axis for validation loss
ax_val = Axis(
    fig[1, 2],
    title="Validation",
    xlabel="epoch",
    ylabel="loss",
    yscale=log10,
)

# Loop through split fractions
for (i, split_frac) in enumerate(sort(collect(keys(loss_train_dict))))
    # Plot training loss
    lines!(
        ax_train,
        loss_train_dict[split_frac],
        label="$(split_frac)",
        color=ColorSchemes.Blues_9[i],
    )
    # Plot validation loss
    lines!(
        ax_val,
        loss_val_dict[split_frac],
        label="$(split_frac)",
        color=ColorSchemes.Blues_9[i],
    )
end # end loop through split fractions

# Add legend
Legend(fig[1, 3], ax_train, framevisible=false)

# Save figure
save("$(fig_dir)/loss_vectors.pdf", fig)

# Display figure
fig

## =============================================================================

println("Plotting loss as a function of split fraction...")

# Initialize figure
fig = Figure(size=(350, 250))

# Extract split fractions
split_fracs = sort(collect(keys(loss_train_dict)))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="split fraction",
    ylabel="loss",
    xticks=split_fracs,
)

# Extract last loss value
loss_train_last = [loss_train_dict[split_frac][end] for split_frac in split_fracs]
loss_val_last = [loss_val_dict[split_frac][end] for split_frac in split_fracs]

# Plot training loss
scatterlines!(ax, split_fracs, loss_train_last, label="training")
# Plot validation loss
scatterlines!(ax, split_fracs, loss_val_last, label="validation")

# Add legend
axislegend(ax, position=:ct)

# Save figure
save("$(fig_dir)/loss_vs_split_frac.pdf", fig)

fig

## =============================================================================

println("Loading neural network...")

# Load neural network
mlp = JLD2.load("$(mlp_dir)/latent_to_phenotype.jld2")["mlp"]

## =============================================================================

println("Plotting predicted phenotypic coordinates...")

# Define number of rows and columns
rows = 2
cols = 4

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows))

# Loop through split fractions
for (i, split_frac) in enumerate(sort(collect(keys(loss_train_dict))))
    # Define figure row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    if i ≠ 1
        # Add axis
        ax = Axis(
            fig[row, col],
            title="split=$(split_frac)",
            aspect=AxisAspect(1),
        )
        # Extract data
        data = JLD2.load(mlp_files[i])["data"]
        # Extract neural network state
        mlp_state = JLD2.load(mlp_files[i])["mlp_state"]
        # Load neural network
        Flux.loadmodel!(mlp, mlp_state)
        # Map latent space coordinates to phenotype space. NOTE: This includes
        # standardizing the input data to have mean zero and standard deviation one
        # and then transforming the output data back to the original scale.
        dd_mlp = DD.DimArray(
            DD.mapslices(
                slice -> StatsBase.reconstruct(
                    data.transforms.x,
                    mlp(StatsBase.transform(data.transforms.z, Vector(slice))),
                ),
                dd_latent,
                dims=:latent,
            ),
            (
                DD.dims(fitnotype_profiles.phenotype)[1],
                dd_latent.dims[2:end]...,
            ),
        )

        # Remove decorations
        hidedecorations!(ax)

        # Plot scatter for predicted phenotype coordinates
        DD.mapslices(
            slice -> scatter!(
                ax,
                Point2f.(eachcol(slice)),
                markersize=5,
                color=ColorSchemes.seaborn_colorblind[2],
            ),
            dd_mlp,
            dims=:phenotype,
        )
    else
        # Add axis
        ax = Axis(
            fig[row, col],
            title="ground truth",
            aspect=AxisAspect(1),
        )
        # Remove decorations
        hidedecorations!(ax)
        # Plot scatter for ground truth phenotype coordinates
        # Plot scatter for phenotype coordinates
        DD.mapslices(
            slice -> scatter!(
                ax,
                Point2f.(eachcol(slice)),
                markersize=5,
                color=ColorSchemes.seaborn_colorblind[1],
            ),
            fitnotype_profiles.phenotype[landscape=DD.At(1)],
            dims=:phenotype,
        )
    end # if
end # end loop through split fractions

# Save figure
save("$(fig_dir)/predicted_phenotype_coordinates.pdf", fig)

fig
