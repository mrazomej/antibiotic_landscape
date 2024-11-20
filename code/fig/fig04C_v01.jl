## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

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

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import StatsBase
import Random

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Define version directory
version_dir = "$(git_root())/output/metropolis-hastings_sim/v05"

# Define simulation directory
sim_dir = "$(version_dir)/sim_evo"
# Define VAE directory
vae_dir = "$(version_dir)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define directory for neural network
mlp_dir = "$(version_dir)/mlp"
# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
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

# List neural network files
mlp_files = sort(Glob.glob("$(mlp_dir)/*split*.jld2"[2:end], "/"))

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

println("Loading neural network...")

# Load neural network
mlp = JLD2.load("$(mlp_dir)/latent_to_phenotype.jld2")["mlp"]

## =============================================================================

println("Plotting figure...")

# Initialize figure
fig = Figure(size=(450, 300))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for banner
gl04C_banner = GridLayout(gl[1, 1])

# Add grid layout for plots
gl04C = GridLayout(gl[2, 1])

# ------------------------------------------------------------------------------
# Section banner
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl04C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50) # Moves box to the left and right
)

# Add section title
Label(
    gl04C_banner[1, 1],
    "comparison of ground truth and learned phenotypic coordinates",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
)

# ------------------------------------------------------------------------------
# Ground truth phenotypic coordinates
# ------------------------------------------------------------------------------

println("Plotting ground truth phenotypic coordinates...")

# Add axis for ground truth phenotypic coordinates
ax_pheno = Axis(
    gl04C[1, 1],
    title="Ground truth\nphenotypic coordinates",
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    xticklabelsvisible=false,
    yticklabelsvisible=false
)

# Plot scatter for phenotype coordinates
DD.mapslices(
    slice -> scatter!(
        ax_pheno,
        Point2f.(eachcol(slice)),
        markersize=5,
        color=ColorSchemes.seaborn_colorblind[1],
    ),
    fitnotype_profiles.phenotype[landscape=DD.At(1)],
    dims=:phenotype,
)

# ------------------------------------------------------------------------------
# Predicted phenotypic coordinates
# ------------------------------------------------------------------------------

println("Plotting predicted phenotypic coordinates...")

# Add axis for predicted phenotypic coordinates
ax_pred = Axis(
    gl04C[1, 2],
    title="Learned\nphenotypic coordinates",
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    xticklabelsvisible=false,
    yticklabelsvisible=false
)

# Extract data
data = JLD2.load(mlp_files[end])["data"]
# Extract neural network state
mlp_state = JLD2.load(mlp_files[end])["mlp_state"]
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

# Plot scatter for predicted phenotype coordinates
DD.mapslices(
    slice -> scatter!(
        ax_pred,
        Point2f.(eachcol(slice)),
        markersize=5,
        color=ColorSchemes.seaborn_colorblind[2],
    ),
    dd_mlp,
    dims=:phenotype,
)

# Save figure
save("$(fig_dir)/fig04C.pdf", fig)

fig
