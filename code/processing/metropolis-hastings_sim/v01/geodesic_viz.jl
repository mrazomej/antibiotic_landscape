## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

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
    # Extract GRN id from file name using regular expression
    id_num = parse(Int, match(r"id(\d+)", gf).captures[1])
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
            :id => id_num,
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

## =============================================================================

println("Loading data into memory...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Reshape the array to stack the 3rd dimension
fit_mat = log.(
    reshape(fitnotype_profiles.fitness.data, size(fitnotype_profiles, 4), :)
)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    reduce(
        (x, y) -> cat(x, y, dims=3),
        StatsBase.transform.(
            Ref(dt), eachslice(log.(fitnotype_profiles.fitness.data), dims=3)
        )
    ),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    rhvae.vae.encoder(log_fitnotype_std.data).μ,
    (latent, log_fitnotype_std.dims[2:end]...)
)

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

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
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)


## =============================================================================

# Define number of columns
cols = 4
# Define the number of needed rows
rows = ceil(Int, length(DD.dims(dd_latent, :lineage)) / cols)

# Initialize figure
fig = Figure(size=(200 * cols, 200 * rows))
# Add grid layout
gl = fig[1, 1] = GridLayout()

# Loop through meta chunks
for i in 1:length(DD.dims(dd_latent, :lineage))
    println("   - Plotting geodesic: $(i)")
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl[row, col],
        aspect=AxisAspect(1),
        title="lineage $(i)",
        xticksvisible=false,
        yticksvisible=false,
    )
    # Hide axis labels
    hidedecorations!(ax)

    # Plot heatmap of log determinant of metric tensor
    heatmap!(
        ax,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=ColorSchemes.tokyo,
    )

    # Plot lineage
    scatterlines!(
        ax,
        dd_latent[latent=DD.At(:latent1), lineage=i].data,
        dd_latent[latent=DD.At(:latent2), lineage=i].data,
        markersize=6,
        linewidth=2,
    )

    # Load geodesic state
    geo_state = JLD2.load(first(df_meta[(df_meta.id.==i), :geodesic_state]))
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
    # Add geodesic line to axis
    lines!(
        ax,
        eachrow(curve)...,
        linewidth=2,
        linestyle=(:dot, :dense),
        color=:white,
    )

    # Add first point 
    scatter!(
        ax,
        [dd_latent[latent=DD.At(:latent1), lineage=i].data[1]],
        [dd_latent[latent=DD.At(:latent2), lineage=i].data[1]],
        color=:white,
        markersize=11,
        marker=:xcross
    )
    scatter!(
        ax,
        [dd_latent[latent=DD.At(:latent1), lineage=i].data[1]],
        [dd_latent[latent=DD.At(:latent2), lineage=i].data[1]],
        color=:black,
        markersize=7,
        marker=:xcross
    )

    # Add last point
    scatter!(
        ax,
        [dd_latent[latent=DD.At(:latent1), lineage=i].data[end]],
        [dd_latent[latent=DD.At(:latent2), lineage=i].data[end]],
        color=:white,
        markersize=11,
        marker=:utriangle
    )
    scatter!(
        ax,
        [dd_latent[latent=DD.At(:latent1), lineage=i].data[end]],
        [dd_latent[latent=DD.At(:latent2), lineage=i].data[end]],
        color=:black,
        markersize=7,
        marker=:utriangle
    )
end # for i in 1:length(DD.dims(dd_latent, :lineage))

# Save figure
save("$(fig_dir)/geodesic_latent_trajectory.pdf", fig)
save("$(fig_dir)/geodesic_latent_trajectory.png", fig)

fig

## =============================================================================

println("Done!")