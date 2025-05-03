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

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Load Plotting packages
using WGLMakie
import ColorSchemes
import Colors
# Activate backend
WGLMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Find the path prefix where to store output
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match, "processing" => ""
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Define model directory
model_dir = "$(git_root())/output$(out_prefix)/model_state"

# Define directory to store trained geodesic curves
geodesic_dir = "$(git_root())/output$(out_prefix)/geodesic_state/"

# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
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
    day_init = parse(Int, match(r"dayinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    day_final = parse(Int, match(r"dayfinal(\d+)", gf).captures[1])
    # Extract evo stress number from file name using regular expression
    env = match(r"evoenv(\w+)_id", gf).captures[1]
    # Extract GRN id from file name using regular expression
    strain_num = parse(Int, match(r"id(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :day_init => day_init,
            :day_final => day_final,
            :env => env,
            :strain_num => strain_num,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

# Sort dataframe by environment
DF.sort!(df_meta, :env)

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Load RHVAE model
rhvae = JLD2.load("$(out_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(model_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading IC50 data...")

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for data in df_group
    # Sort data by drug
    DF.sort!(data, :drug)
    # Run :logic50_mean_std through encoder
    latent = rhvae.vae.encoder(data.logic50_mean_std).µ
    # Append latent coordinates to dataframe
    DF.append!(
        df_latent,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent[1, :],
            :latent2 => latent[2, :],
            :latent3 => latent[3, :],
        )
    )
end # for 

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent.latent1) - 0.5,
    maximum(df_latent.latent1) + 0.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent.latent2) - 0.5,
    maximum(df_latent.latent2) + 0.5,
    length=n_points
)
latent3_range = range(
    minimum(df_latent.latent3) - 0.5,
    maximum(df_latent.latent3) + 0.5,
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

# Extract all unique pairs
unique_pairs = unique(df_meta[:, [:env, :strain_num]])

# Define (:env, :strain_num) pairs to plot
pairs = [
    ("KM", 16), ("NFLX", 33), ("TET", 8), ("KM", 28), ("NFLX", 35), ("TET", 3)
]

# Select example pair
p = pairs[5]

# p = collect(unique_pairs[55, :])

println(p)

# Extract latent coordinates
df_latent_subset = df_latent[
    (df_latent.env.==p[1]).&(df_latent.strain_num.==p[2]),
    :
]

# Load geodesic state
geo_state = JLD2.load(
    first(
        df_meta[
            (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]),
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

# Plot geodesic line
lines!(
    ax,
    eachrow(curve)...,
    linewidth=2,
    color=:black
)

# Plot data trajectory
scatterlines!(
    ax,
    df_latent_subset.latent1,
    df_latent_subset.latent2,
    df_latent_subset.latent3,
    markersize=8,
    linewidth=2,
    color=:red,
)

fig

