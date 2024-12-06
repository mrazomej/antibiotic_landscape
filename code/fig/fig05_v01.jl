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
import CSV

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import LinearAlgebra
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

# Define model directory
model_dir = "$(git_root())/output/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/v05"
# Define state directory
state_dir = "$(model_dir)/model_state"


# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading MCMC summary results...")

# Load results
df = CSV.File(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_ci.csv"
) |> DF.DataFrame

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
# Static Fig05
## =============================================================================

# Set random seed
Random.seed!(42)

# Define evolution drug
evo_drug = "TET"

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(600, 600))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig05A section banner
gl05A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig05A
gl05A = gl[2, 1] = GridLayout()

# Add grid layout for fig05B section banner
gl05B_banner = gl[3, 1] = GridLayout()
# Add grid layout for Fig05B
gl05B = gl[4, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust row sizes
rowsize!(gl, 2, Auto(3))
rowsize!(gl, 4, Auto(2))

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl05A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-45, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05A_banner[1, 1],
    "IC₅₀ profiles for strains evolved in $(evo_drug)",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-35) # Moves text to the left
)

# Add box
box = Box(
    gl05B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-45, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05B_banner[1, 1],
    "PCA vs. RHVAE latent space comparison",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-35) # Moves text to the left
)

# ------------------------------------------------------------------------------
# Fig05A
# ------------------------------------------------------------------------------

# Define number of rows and columns
rows = 2
cols = 4

# Find index of entries where :env contain "in_$(evo_drug)"
idx = findall(x -> occursin("in_$(evo_drug)", x), df.env)

# Group data by drug
df_group = DF.groupby(df[idx, :], :drug)

# Convert GroupedDataFrame to Vector and reorder so matching drug comes first
df_group_vec = collect(df_group)
sort!(df_group_vec, by=x -> x[1, :drug] != evo_drug)

# Define colors for each strain as a dictionary
colors = Dict(
    sort(unique(df[idx, :strain_num])) .=>
        ColorSchemes.glasbey_hv_n256[1:length(unique(df[idx, :strain_num]))]
)

# Loop through each drug
for (i, data) in enumerate(df_group_vec)
    # Define index for row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl05A[row, col],
        xlabel="day",
        ylabel="log(IC₅₀)",
        title=(data[1, :drug] == evo_drug) ?
              "$(data[1, :drug]) (selection)" :
              "$(data[1, :drug])",
        titlesize=12,
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=12,
        yticklabelsize=12,
        aspect=AxisAspect(4 / 3),
    )

    # Group data by :strain_num
    data_group = DF.groupby(data, [:strain_num])

    # Loop through each strain
    for (j, strain) in enumerate(data_group)
        # Sort data by day
        DF.sort!(strain, :day)
        # Extract strain number
        strain_num = first(strain.strain_num)
        # Plot data
        scatterlines!(
            strain.day,
            strain.logic50_mean,
            color=colors[strain_num],
            markersize=4,
        )
    end # for
end # for

# Adjust gap between rows and columns
rowgap!(gl05A, 5)
colgap!(gl05A, 5)

# ------------------------------------------------------------------------------
# Fig05B
# ------------------------------------------------------------------------------

# Add axis for PCA
ax_pca = Axis(
    gl05B[1, 1],
    title="PCA",
    xlabel="PC1",
    ylabel="PC2",
    titlesize=12,
    xlabelsize=12,
    ylabelsize=12,
    xticklabelsize=12,
    yticklabelsize=12,
    aspect=AxisAspect(1),
)

# Add axis for latent space
ax_latent = Axis(
    gl05B[1, 2],
    title="RHVAE",
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    titlesize=12,
    xlabelsize=12,
    ylabelsize=12,
    xticklabelsize=12,
    yticklabelsize=12,
    aspect=AxisAspect(1),
)

# Add axis for reconstruction error
ax_mse = Axis(
    gl05B[1, 3],
    title="reconstruction error",
    xlabel="# of PCA components",
    ylabel="mean squared error",
    xticks=0:2:n_env,
    titlesize=12,
    xlabelsize=12,
    ylabelsize=12,
    xticklabelsize=12,
    yticklabelsize=12,
    aspect=AxisAspect(1),
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

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

# Add subplot labels
Label(
    gl05A[1, 1, TopLeft()], "(A)",
    fontsize=20,
    padding=(0, 20, 30, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl05B[1, 1, TopLeft()], "(B)",
    fontsize=20,
    padding=(0, 20, 30, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Save figure
save("$(fig_dir)/fig05_v01.png", fig)
save("$(fig_dir)/fig05_v01.pdf", fig)

fig
