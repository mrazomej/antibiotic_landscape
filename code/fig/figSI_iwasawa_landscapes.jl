## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DataFrames as DF
import CSV

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import LinearAlgebra
import MultivariateStats as MStats
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

# Define model directory
model_dir = "$(git_root())/output/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/v05"
# Define state directory
rhvae_state_dir = "$(model_dir)/model_state"

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading IC50 data...")

# Load logic50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load list of drugs
drug_list = sort(unique(df_logic50.drug))
# Load standardize data
data_std = JLD2.load("$(data_dir)/logic50_preprocess.jld2")["logic50_mean_std"]

## =============================================================================

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
            :model .=> :rhvae
        )
    )
end # for 

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 300

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent[df_latent.model.==:rhvae, :latent1]) - 1.5,
    maximum(df_latent[df_latent.model.==:rhvae, :latent1]) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent[df_latent.model.==:rhvae, :latent2]) - 1.5,
    maximum(df_latent[df_latent.model.==:rhvae, :latent2]) + 1.5,
    length=n_points
)
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Decode latent points
ic50_rhvae = reshape(
    rhvae.vae.decoder(z_mat).µ',
    n_points,
    n_points,
    length(drug_list)
)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

## =============================================================================

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.92 .< logdetG .<= maximum(logdetG))

# Initialize plot
fig = Figure(size=(700, 600))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add grid layout for banner
gl_banner = gl[1, 1] = GridLayout()

# Add grid layout for subplots
gl_subplots = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Add banner
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-40, right=-40), # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "fitness landscapes in latent space from antibiotic resistance data",
    fontsize=12,
    padding=(-30, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# ------------------------------------------------------------------------------
# Add subplots
# ------------------------------------------------------------------------------

# Define the number of rows and columns
rows = 3
cols = 3

# Loop through environments
for (i, fit_landscape) in enumerate(eachslice(ic50_rhvae, dims=3))
    # Mask fitness landscape
    fit_landscape_masked = (mask .* minimum(fit_landscape)) .+
                           (fit_landscape .* .!mask)

    # Calculate row and column indices
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1

    # Add GridLayout to insert axis
    gl_ax = gl_subplots[row, col] = GridLayout()
    # Add axis to plot
    ax = Axis(
        gl_ax[1, 1],
        xlabel="latent dimension 1",
        ylabel="latent dimension 2",
        aspect=AxisAspect(1),
        title=drug_list[i],
    )
    # Remove axis labels
    hidedecorations!(ax)
    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax,
        latent1_range,
        latent2_range,
        fit_landscape_masked,
        colormap=:algae,
        rasterize=true,
        colorrange=(
            minimum(fit_landscape), maximum(fit_landscape)
        ),
    )

    # Add contour lines
    contour!(
        ax,
        latent1_range,
        latent2_range,
        fit_landscape_masked,
        color=:black,
        linestyle=:dash,
        levels=7,
    )

    # Add colorbar
    cb = Colorbar(
        gl_ax[1, 2],
        hm,
        size=8,
        label="log(IC₅₀)",
        labelsize=12,
        labelpadding=0.0,
        ticklabelsize=12,
        ticksvisible=false,
        tellheight=false,
        tellwidth=true,
        height=132.5,
    )

end # for

# ------------------------------------------------------------------------------
# Curvature plot
# ------------------------------------------------------------------------------

# Add extra GridLayout to insert axis
gl_curvature = gl_subplots[3, 3] = GridLayout()

# Add extra axis to plot
ax = Axis(
    gl_curvature[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="metric volume",
    aspect=AxisAspect(1),
)
# Remove axis labels
hidedecorations!(ax)

# Plot heatmap of log determinant of metric tensor
hm = heatmap!(
    ax,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
    rasterize=true,
)
# Add colorbar
cb = Colorbar(
    gl_curvature[1, 2],
    hm,
    size=8,
    label="log√det(G)",
    labelsize=12,
    labelpadding=0.0,
    ticklabelsize=12,
    ticksvisible=false,
    tellheight=false,
    tellwidth=true,
    height=132.5,
)

# ------------------------------------------------------------------------------
# Add labels
# ------------------------------------------------------------------------------

Label(
    gl_subplots[end, :, Bottom()],
    "latent dimension 1",
    fontsize=16,
    padding=(0, 0, 0, 10),
)
Label(
    gl_subplots[2, :, Left()],
    "latent dimension 2",
    fontsize=16,
    rotation=π / 2,
    padding=(0, 10, 0, 0),
)

# Save figure
save("$(fig_dir)/figSI_iwasawa_landscapes.pdf", fig)
save("$(fig_dir)/figSI_iwasawa_landscapes.png", fig)

fig