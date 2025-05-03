## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

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

# Activate backend
WGLMakie.activate!()
# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

Random.seed!(42)

## =============================================================================

# Locate current directory
path_dir = pwd()

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
# Define directory to store trained geodesic curves
geodesic_dir = "$(vae_dir)/geodesic_state/"
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/vae"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    t_init = parse(Int, match(r"timeinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    t_final = parse(Int, match(r"timefinal(\d+)", gf).captures[1])
    # Extract lineage number from file name using regular expression
    lin = parse(Int, match(r"lineage(\d+)", gf).captures[1])
    # Extract replicate number from file name using regular expression
    rep = parse(Int, match(r"replicate(\d+)", gf).captures[1])
    # Extract evolution condition from file name using regular expression
    evo = parse(Int, match(r"evo(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :t_init => t_init,
            :t_final => t_final,
            :lineage => lin,
            :rep => rep,
            :evo => evo,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(vae_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Load RHVAE model
rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

# Extract latent space dimensionality
ldim = size(rhvae.vae.encoder.μ.weight, 1)

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
latent_ranges = Dict(
    :latent1 => latent1_range, 
    :latent2 => latent2_range, 
    :latent3 => latent3_range
)

# Define grid points
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

println("Plotting example geodesic curves with 3D volume slices in background...")

# ------------------------------------------------------------------------------
# Geodesic loading
# ------------------------------------------------------------------------------
# Define subset of data
subset = (rep=1, evo=1, lineage=1)

# Select subset of data
dd_subset = dd_latent[
    replicate=subset.rep,
    evo=subset.evo,
    lineage=subset.lineage,
]

# Load geodesic state
geo_state = JLD2.load(
    first(
        df_meta[
            (df_meta.rep.==subset.rep).&(df_meta.evo.==subset.evo).&(df_meta.lineage.==subset.lineage),
            :geodesic_state]
    )
)

# Define NeuralGeodesic model
nng = NG.NeuralGeodesic(
    nng_template,
    geo_state["latent_init"],
    geo_state["latent_end"],
)
# Update model state
Flux.loadmodel!(nng, geo_state["model_state"])
# Generate curve
curve = nng(t_array)

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
# Initialize figure
fig = Figure(size=(600, 600))
# Add axis as a 3D scene
ax = LScene(
    fig[1, 1],
    show_axis=false,
)

# Add data trajectory to axis
scatterlines!(
    ax,
    Point3f.(eachcol(dd_subset.data)),
    markersize=5,
)

# Add geodesic curve to axirows
lines!(
    ax,
    Point3f.(eachcol(curve)),
    linewidth=2,
    color=:white,
    # linestyle=(:dot, :dense),
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
    alpha=true
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

fig

## =============================================================================

# Define subset of data (now only fixing replicate and evolution condition)
subset = (rep=1, evo=1)

# Get unique lineages for this replicate and evolution condition
lineages = unique(
    df_meta[(df_meta.rep.==subset.rep).&(df_meta.evo.==subset.evo), :lineage]
)

# Initialize figure
fig = Figure(size=(600, 600))
# Add axis as a 3D scene
ax = LScene(
    fig[1, 1],
    show_axis=false,
)

# Loop through each lineage
for lineage in lineages
    # Select subset of data for this lineage
    dd_subset = dd_latent[
        replicate=subset.rep,
        evo=subset.evo,
        lineage=lineage,
    ]

    # Load geodesic state for this lineage
    geo_state = JLD2.load(
        first(
            df_meta[
                (df_meta.rep.==subset.rep).&(df_meta.evo.==subset.evo).&(df_meta.lineage.==lineage),
                :geodesic_state
            ]
        )
    )

    # Define NeuralGeodesic model
    nng = NG.NeuralGeodesic(
        nng_template,
        geo_state["latent_init"],
        geo_state["latent_end"],
    )
    # Update model state
    Flux.loadmodel!(nng, geo_state["model_state"])
    # Generate curve
    curve = nng(t_array)

    # Add data trajectory to axis
    scatterlines!(
        ax,
        Point3f.(eachcol(dd_subset.data)),
        markersize=5,
    )

    # Add geodesic curve
    lines!(
        ax,
        Point3f.(eachcol(curve)),
        linewidth=2,
        color=:white,
    )
end

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
    alpha=true
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

fig