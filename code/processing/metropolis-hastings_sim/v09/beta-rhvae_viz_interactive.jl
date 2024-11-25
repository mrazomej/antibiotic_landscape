## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import Flux for training VAEs
import Flux

# Import libraries to handle data
import Glob
import DimensionalData as DD
import DataFrames as DF

# Import JLD2 for saving results
import JLD2

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random
import Distributions
import Distances
# Load CairoMakie for plotting
using WGLMakie
using Bonito
import ColorSchemes
import PDFmerger: append_pdf!

# Activate backend
WGLMakie.activate!()
# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

Random.seed!(42)

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
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/vae"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = Glob.glob("$(state_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over files
for f in model_states
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
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for f in model_states

## =============================================================================

println("Loading data into memory...")

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

println("Load model...")

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(df_meta.model_state[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

# Extract latent space dimensionality
ldim = size(rhvae.vae.encoder.μ.weight, 1)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}(Symbol.(["latent$x" for x in 1:ldim]))

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

# Reorder dimensions to have latent as the first dimension
dd_latent = DD.permutedims(dd_latent, (4, 1, 2, 3, 5))

## =============================================================================

println("Plotting latent space coordinates...")

# Initialize figure
fig = Figure(size=(400, 400))
# Add 3D axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent 1",
    ylabel="latent 2",
    zlabel="latent 3",
    aspect=(1, 1, 1),
)
# Plot scatter using mapslices
mapslices(
    slice -> scatter!(
        ax,
        Point3f(slice),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[1], 0.5)
    ),
    dd_latent,
    dims=:latent,
)

# Save figure
save("$(fig_dir)/rhvae_latent_space.png", fig)

# Display figure
fig

## =============================================================================

println("Plotting latent space coordinates colored by lineage...")

# Initialize figure
fig = Figure(size=(400, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_latent, :lineage))
    mapslices(
        slice -> scatter!(
            ax,
            Point3f(slice),
            markersize=5,
            color=(ColorSchemes.glasbey_hv_n256[i], 0.25)
        ),
        dd_latent[lineage=lin],
        dims=:latent,
    )
end # for 

# Save figure
save("$(fig_dir)/rhvae_latent_space_lineage.png", fig)

fig

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Initialize array to store ranges for each latent dimension
latent_ranges = Dict{Symbol,Vector{Float32}}()

# Loop through latent dimensions
for latent_dim in DD.dims(dd_latent, :latent)
    # Extract range for this dimension
    latent_ranges[latent_dim] = range(
        minimum(dd_latent[latent=DD.At(latent_dim)]) - 1.5,
        maximum(dd_latent[latent=DD.At(latent_dim)]) + 1.5,
        length=n_points
    )
end

# Create grid points for all dimensions
grid_points = Iterators.product(values(latent_ranges)...)

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

println("Plotting latent space metric as a 3D volume slice...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis as a 3D scene
ax = LScene(
    fig[1, 1],
    show_axis=false,
)

# Add sliders
sgrid = SliderGrid(
    fig[2, 1],
    (label="yz plane - x axis", range=1:length(latent_ranges[:latent1])),
    (label="xz plane - y axis", range=1:length(latent_ranges[:latent2])),
    (label="xy plane - z axis", range=1:length(latent_ranges[:latent3])),
)

# Extract layout 
lo = sgrid.layout
# Extract number of columns
nc = ncols(lo)

# Plot volume slices
plt = volumeslices!(
    ax,
    values(latent_ranges)...,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

# Extract sliders
sl_yz, sl_xz, sl_xy = sgrid.sliders

# Connect sliders to `volumeslices` update methods  
on(sl_yz.value) do v
    plt[:update_yz][](v)
end
on(sl_xz.value) do v
    plt[:update_xz][](v)
end
on(sl_xy.value) do v
    plt[:update_xy][](v)
end

# Set sliders to close to the middle of the range
set_close_to!(sl_yz, 0.5 * length(latent_ranges[:latent1]))
set_close_to!(sl_xz, 0.5 * length(latent_ranges[:latent2]))
set_close_to!(sl_xy, 0.5 * length(latent_ranges[:latent3]))

# Save figure
save("$(fig_dir)/rhvae_latent_space_metric.png", fig)

fig

## =============================================================================

println("Plotting latent space metric as a 3D volume slice with data scatter...")

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis as a 3D scene
ax = LScene(
    fig[1, 1],
    show_axis=false,
)

# Plot scatter using mapslices
mapslices(
    slice -> scatter!(
        ax,
        Point3f(slice),
        markersize=5,
        color=(:white, 0.05)
    ),
    dd_latent,
    dims=:latent,
)

# Add sliders
sgrid = SliderGrid(
    fig[2, 1],
    (label="yz plane - x axis", range=1:length(latent_ranges[:latent1])),
    (label="xz plane - y axis", range=1:length(latent_ranges[:latent2])),
    (label="xy plane - z axis", range=1:length(latent_ranges[:latent3])),
)

# Extract layout 
lo = sgrid.layout
# Extract number of columns
nc = ncols(lo)

# Plot volume slices
plt = volumeslices!(
    ax,
    values(latent_ranges)...,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

# Extract sliders
sl_yz, sl_xz, sl_xy = sgrid.sliders

# Connect sliders to `volumeslices` update methods  
on(sl_yz.value) do v
    plt[:update_yz][](v)
end
on(sl_xz.value) do v
    plt[:update_xz][](v)
end
on(sl_xy.value) do v
    plt[:update_xy][](v)
end

# Set sliders to close to the middle of the range
set_close_to!(sl_yz, 0.5 * length(latent_ranges[:latent1]))
set_close_to!(sl_xz, 0.5 * length(latent_ranges[:latent2]))
set_close_to!(sl_xy, 0.5 * length(latent_ranges[:latent3]))

# Save figure
save("$(fig_dir)/rhvae_latent_space_metric_scatter.png", fig)

fig

## =============================================================================

println("Plotting latent space metric as contour...")

# Initialize figure
fig = Figure(size=(400, 400))
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
    values(latent_ranges)...,
    logdetG,
    alpha=0.05,
    levels=7,
    colormap=ColorSchemes.tokyo,
)

# Save figure
save("$(fig_dir)/rhvae_latent_space_metric_contour.png", fig)

fig

## =============================================================================

