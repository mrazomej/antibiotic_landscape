## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import CSV
import DataFrames as DF
import DimensionalData as DD

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Import WGLMakie for interactive plotting
using GLMakie
import ColorSchemes
import Colors
# Activate backend
GLMakie.activate!()
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
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading data into memory...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Define by how much to subsample the time series
n_sub = 10

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

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2, :latent3])

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

## =============================================================================

# Initialize figure
fig = Figure(size=(500, 500))

# Define layout
gl = fig[1, 1] = GridLayout()

# Add 3D axis
ax = Axis3(
    gl[1, 1],
    aspect=(1, 1, 1),
    xgridvisible=false,
    ygridvisible=false,
    zgridvisible=false,
)

mapslices(
    slice -> scatter!(ax, slice..., color=ColorSchemes.seaborn_colorblind[1]),
    dd_latent[replicate=DD.At(1)].data,
    dims=[3],
)

fig

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 50

# Extract latent space ranges
latent1_range = range(
    minimum(dd_latent[latent=DD.At(:latent1)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent1)]) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(dd_latent[latent=DD.At(:latent2)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent2)]) + 1.5,
    length=n_points
)
latent3_range = range(
    minimum(dd_latent[latent=DD.At(:latent3)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent3)]) + 1.5,
    length=n_points
)

# Define latent ranges
latent_ranges = (latent1_range, latent2_range, latent3_range)

# Define grid points
grid_points = Iterators.product(latent_ranges...)

# Compute inverse metric tensor
Ginv = map(
    point -> AET.RHVAEs.G_inv([point...], rhvae),
    grid_points
)

# Compute metric 
logdetG = map(
    Ginv -> -1 / 2 * AET.utils.slogdet(Ginv),
    Ginv
)

## =============================================================================

println("Plotting latent space metric as contour...")

# Initialize figure
fig = Figure(size=(500, 500))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Plot contour
contour!(
    ax,
    collect(latent_ranges[1]),
    collect(latent_ranges[2]),
    collect(latent_ranges[3]),
    logdetG,
    alpha=0.05,
    levels=7,
    colormap=ColorSchemes.tokyo,
)

fig
