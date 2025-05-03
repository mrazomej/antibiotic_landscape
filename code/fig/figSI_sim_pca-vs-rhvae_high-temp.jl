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
import LinearAlgebra

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
version_dir = "$(git_root())/output/metropolis-hastings_sim/v07"

# Define simulation directory
sim_dir = "$(version_dir)/sim_evo"
# Define VAE directory
vae_dir = "$(version_dir)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define directory for neural network
mlp_dir = "$(version_dir)/mlp"
# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

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

println("Loading RHVAE model...")

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

println("Performing PCA on fitness data...")

# Standardize fit_mat to have mean zero and standard deviation one
fit_mat_std = StatsBase.transform(
    StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2),
    fit_mat
)

# Perform SVD on the data
fit_svd = LinearAlgebra.svd(fit_mat_std)

# Extract principal components
pcs = fit_svd.U

## =============================================================================

println("Computing PCA mean squared error of the data reconstruction...")

# Compute mean squared error of the data reconstruction
pca_mse = [
    begin
        # Compute the reconstruction of the data using U, Σ, and Vᵀ
        fit_mat_recon = fit_svd.U[:, 1:i] *
                        LinearAlgebra.Diagonal(fit_svd.S[1:i]) *
                        fit_svd.Vt[1:i, :]
        # Compute the mean squared error
        Flux.mse(fit_mat_std, fit_mat_recon)
    end for i in 1:n_env
]


## =============================================================================

println("Projecting data to the first two principal components...")

# Define principal components dimensions
pca = DD.Dim{:pca}([:pc1, :pc2])

# Map data to principal components
dd_pca = DD.DimArray(
    dropdims(
        mapslices(slice -> pcs[:, 1:2]' * slice,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., pca, log_fitnotype_std.dims[6]),
)

# Reorder dimensions
dd_pca = permutedims(dd_pca, (4, 1, 2, 3, 5))

## =============================================================================

println("Plotting PCA and latent space comparison...")

# Initialize figure
fig = Figure(size=(600, 600))

# -----------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add banner grid layout
gl_banner = GridLayout(gl[1, 1])

# Add plot grid layout
gl_plot = GridLayout(gl[2, 1])

# -----------------------------------------------------------------------------

# Add box
box = Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-55, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "PCA vs. RHVAE latent space comparison",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-40) # Moves text to the left
)

# -----------------------------------------------------------------------------

# Add axis for true phenotypic coordinates
ax_pheno = Axis(
    gl_plot[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Phenotype space"
)

# Add axis for PCA
ax_pca = Axis(
    gl_plot[1, 2],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA space"
)

# Add axis for latent space
ax_latent = Axis(
    gl_plot[2, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE latent space"
)

# Add axis for reconstruction error
ax_mse = Axis(
    gl_plot[2, 2],
    xlabel="# of PCA components",
    ylabel="mean squared error",
    yscale=log10,
    aspect=AxisAspect(1),
    title="Reconstruction error"
)

# -----------------------------------------------------------------------------

# Group data by :lineage
dd_pheno_group = DD.groupby(
    fitnotype_profiles.phenotype[landscape=DD.At(1)],
    DD.dims(fitnotype_profiles.phenotype, :lineage),
)

# Loop through groups
for (i, data) in enumerate(dd_pheno_group)
    # Extract the data and reshape it to 2×N matrix
    points = reshape(data.data, 2, :)
    # Plot the points
    scatter!(
        ax_pheno,
        eachrow(points)...,
        markersize=5,
        color=ColorSchemes.glasbey_hv_n256[i],
        rasterize=true,
    )
end

# Group data by :lineage
dd_pca_group = DD.groupby(
    dd_pca,
    DD.dims(dd_pca, :lineage),
)

# Loop through groups
for (i, data) in enumerate(dd_pca_group)
    # Extract the data and reshape it to 2×N matrix
    points = reshape(data.data, 2, :)
    scatter!(
        ax_pca,
        eachrow(points)...,
        markersize=5,
        color=ColorSchemes.glasbey_hv_n256[i],
        rasterize=true,
    )
end

# Group data by :lineage
dd_latent_group = DD.groupby(
    dd_latent,
    DD.dims(dd_latent, :lineage),
)

# Loop through groups
for (i, data) in enumerate(dd_latent_group)
    # Extract the data and reshape it to 2×N matrix
    points = reshape(data.data, 2, :)
    # Plot the points using modulo to cycle through colors
    scatter!(
        ax_latent,
        eachrow(points)...,
        markersize=5,
        color=ColorSchemes.glasbey_hv_n256[i],
        rasterize=true,
    )
end

# -----------------------------------------------------------------------------

# Define number of principal components to plot
n_pca = 15

# Plot reconstruction error
scatterlines!(
    ax_mse,
    1:n_pca,
    pca_mse[1:n_pca],
    label="PCA",
    color=ColorSchemes.seaborn_colorblind[1],
)

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(fit_mat_std, rhvae(fit_mat_std).µ)

# Plot reconstruction error as horizontal line
hlines!(
    ax_mse,
    mse_rhvae,
    label="2D RHVAE",
    color=ColorSchemes.seaborn_colorblind[2],
    linewidth=2,
)

# Add legend
axislegend(ax_mse, position=:rt, labelsize=12, framevisible=false)

# -----------------------------------------------------------------------------

# Change spacing between subplots
rowgap!(gl_plot, 5)
colgap!(gl_plot, -5)

# Save figure
save("$(fig_dir)/figSI_sim_pca-vs-rhvae_high-temp.png", fig)
save("$(fig_dir)/figSI_sim_pca-vs-rhvae_high-temp.pdf", fig)

fig