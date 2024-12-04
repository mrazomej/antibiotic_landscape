## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Load libraries for machine learning
import AutoEncoderToolkit as AET
import Flux

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import Glob to list files
import Glob

# Import JLD2 to load model
import JLD2

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Define model directory
model_dir = "$(git_root())/output/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/v05"
# Define state directory
state_dir = "$(model_dir)/model_state"

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading data and building standardized matrix...")

# Define data directory
data_dir = "$(git_root())/data/Iwasawa_2022"

# Load file into memory
df_ic50 = CSV.read("$(data_dir)/iwasawa_ic50_tidy.csv", DF.DataFrame)

# Locate strains with missing values
missing_strains = unique(df_ic50[ismissing.(df_ic50.log2ic50), :strain])

# Remove data
df_ic50 = df_ic50[[x ∉ missing_strains for x in df_ic50.strain], :]

# Group data by strain and day
df_group = DF.groupby(df_ic50, [:strain, :day])

# Extract unique drugs to make sure the matrix is built correctly
drug = sort(unique(df_ic50.drug))

# Initialize matrix to save ic50 values
ic50_mat = Matrix{Float32}(undef, length(drug), length(df_group))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Check that the stress are in the correct order
    if all(data.drug .== drug)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.log2ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)
# Define number of samples
n_samples = size(ic50_mat, 2)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat)

## =============================================================================

println("Performing PCA...")

# Perform SVD on the data
ic50_svd = LinearAlgebra.svd(ic50_std)

# Extract principal components
pcs = ic50_svd.U

# Convert singular values to explained variance
ic50_var = ic50_svd.S .^ 2 / (n_samples - 1)

# Compute explained variance percentage
ic50_var_pct = ic50_var / sum(ic50_var)

## =============================================================================

println("Computing PCA mean squared error of the data reconstruction...")

# Compute mean squared error of the data reconstruction
pca_mse = [
    begin
        # Compute the reconstruction of the data
        ic50_recon = pcs[:, 1:i] * (pcs[:, 1:i]' * ic50_std)
        # Compute the mean squared error
        Flux.mse(ic50_std, ic50_recon)
    end for i in 1:n_env
]

## =============================================================================

println("Projecting data to the first two principal components...")

# Project data to the first two principal components
data_pca = pcs[:, 1:2]' * ic50_std

# Convert data to DataFrame
df_pca = DF.DataFrame(
    data_pca',
    [:pc1, :pc2],
)

# Change sign for pc1
df_pca.pc1 = -df_pca.pc1

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Add strains and days to DataFrame
DF.insertcols!(
    df_pca,
    :strain => strains_mat,
    :day => day_mat
)

# Add corresponding metadata resistance value
df_pca = DF.leftjoin!(
    df_pca,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

## =============================================================================

println("Loading RHVAE model...")

# Find model file
model_file = first(Glob.glob("$(model_dir)/model*.jld2"[2:end], "/"))
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

# Project data to RHVAE latent space
data_latent = rhvae.vae.encoder(ic50_std).µ

# Convert data to DataFrame
df_latent = DF.DataFrame(
    data_latent',
    [:z1, :z2],
)

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Add strains and days to DataFrame
DF.insertcols!(
    df_latent,
    :strain => strains_mat,
    :day => day_mat
)

# Add corresponding metadata resistance value
df_latent = DF.leftjoin!(
    df_latent,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

first(df_latent, 5)

## =============================================================================

println("Merging PCA and latent data...")

# Merge PCA and latent data
df_red = DF.leftjoin(df_pca, df_latent, on=[:strain, :day, :parent, :env])

## =============================================================================

println("Plotting PCA and latent space comparison...")

# Initialize figure
fig = Figure(size=(700, 300))

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

# Add axis for PCA
ax_pca = Axis(
    gl_plot[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA"
)

# Add axis for latent space
ax_latent = Axis(
    gl_plot[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE"
)

# Add axis for reconstruction error
ax_mse = Axis(
    gl_plot[1, 3],
    xlabel="# of PCA components",
    ylabel="mean squared error",
)

# Pack first two axes
axes = [ax_pca, ax_latent]

# -----------------------------------------------------------------------------

# Group data by :parent and :env
df_red_group = DF.groupby(df_red, [:parent, :env])

# Loop through groups
for (i, data) in enumerate(df_red_group)
    # Plot data
    scatter!(
        axes[1],
        data.pc1,
        data.pc2,
        markersize=5,
        color=ColorSchemes.glasbey_hv_n256[i],
    )
    scatter!(
        axes[2],
        data.z1,
        data.z2,
        markersize=5,
        color=ColorSchemes.glasbey_hv_n256[i],
    )
end # for

# -----------------------------------------------------------------------------

# Plot reconstruction error
scatterlines!(
    ax_mse,
    1:n_env,
    pca_mse,
    label="PCA",
    color=ColorSchemes.seaborn_colorblind[1],
)

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(ic50_std, rhvae(ic50_std).µ)

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

# Save figure
save("$(fig_dir)/figSI_data_pca-vs-rhvae.png", fig)
save("$(fig_dir)/figSI_data_pca-vs-rhvae.pdf", fig)

fig