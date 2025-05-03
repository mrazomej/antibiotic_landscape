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
rhvae_state_dir = "$(model_dir)/model_state"
vae_state_dir = "$(model_dir)/vae_model_state"


# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading raw data...")

# Load data into a DataFrame
df_raw = CSV.read(
    "$(git_root())/data/Iwasawa_2022/iwasawa_tidy.csv", DF.DataFrame
)

# Remove blank measurements
df_raw = df_raw[.!df_raw.blank, :]
# Remove zero concentrations
df_raw = df_raw[df_raw.concentration_ugmL.>0, :]

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
    [:latent1, :latent2],
)

# Change sign for pc1
df_pca.latent1 = -df_pca.latent1

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

# Add column for latent space
df_pca.latent_space .= "PCA"

## =============================================================================

println("Loading RHVAE model...")

# Find model file
model_file = first(Glob.glob("$(model_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading VAE model...")

# Find model file
model_file = first(Glob.glob("$(model_dir)/vae_model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = sort(Glob.glob("$(vae_state_dir)/*.jld2"[2:end], "/"))

# Load model
vae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(vae, JLD2.load(model_states[end])["model_state"])

## =============================================================================

println("Mapping data to latent space...")

# Project data to RHVAE latent space
data_rhvae = rhvae.vae.encoder(ic50_std).µ
data_vae = vae.encoder(ic50_std).µ

# Convert data to DataFrame
df_rhvae = DF.DataFrame(
    data_rhvae',
    [:latent1, :latent2],
)
df_vae = DF.DataFrame(
    data_vae',
    [:latent1, :latent2],
)

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Add strains and days to DataFrame
DF.insertcols!.(
    [df_rhvae, df_vae],
    :strain => strains_mat,
    :day => day_mat
)

# Add corresponding metadata resistance value
DF.leftjoin!.(
    [df_rhvae, df_vae],
    Ref(unique(df_ic50[:, [:strain, :day, :parent, :env]])),
    on=[:strain, :day]
)

# Add column for latent space
df_rhvae.latent_space .= "RHVAE"
df_vae.latent_space .= "VAE"

first(df_rhvae, 5)

## =============================================================================

println("Merging latent space data...")

# Merge PCA and latent data
df_latent = DF.vcat(df_pca, df_rhvae, df_vae)

## =============================================================================
# Static Fig05
## =============================================================================

# Set random seed
Random.seed!(42)

# Define evolution drug
evo_drug = "TET"

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(700, 800))

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
gl05B_banner = gl[1, 2] = GridLayout()
# Add grid layout for Fig05B
gl05B = gl[2, 2] = GridLayout()

# Add grid layout for fig05C section banner
gl05C_banner = gl[3, 1:2] = GridLayout()
# Add grid layout for Fig05C
gl05C = gl[4, 1:2] = GridLayout()

# Add grid layout for fig05D section banner
gl05D_banner = gl[5, 1:2] = GridLayout()
# Add grid layout for Fig05D
gl05D = gl[6, 1:2] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust row sizes
rowsize!(gl, 2, Auto(2))
rowsize!(gl, 4, Auto(3))
rowsize!(gl, 6, Auto(2))

# Adjust column sizes
colsize!(gl, 1, Auto(4))
colsize!(gl, 2, Auto(3))

# Adjust rowgap
rowgap!(gl, 5, -5)

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl05A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05A_banner[1, 1],
    "schematic of experimental setup",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-5) # Moves text to the left
)

# Add box for section title
Box(
    gl05B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05B_banner[1, 1],
    "antibiotic resistance progression in evolution\ncondition",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    justification=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-5) # Moves text to the left
)

# Add box for section title
Box(
    gl05C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05C_banner[1, 1],
    "IC₅₀ profiles for strains evolved in $(evo_drug)",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-5) # Moves text to the left
)

# Add box
box = Box(
    gl05D_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-15, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05D_banner[1, 1],
    "comparison of learned latent space coordinates",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-5) # Moves text to the left
)

# ------------------------------------------------------------------------------
# Fig05B
# ------------------------------------------------------------------------------

println("Plotting Fig05B...")

# Define data to use
data_raw = df_raw[(df_raw.antibiotic.=="KM").&(df_raw.env.=="Parent_in_KM").&(df_raw.strain_num.==13).&.!(df_raw.blank).&(df_raw.concentration_ugmL.>0), :]
# Group data by day
df_raw_group = DF.groupby(data_raw, :day)

# Add axis
ax = Axis(
    gl05B[1, 1],
    xlabel="[antibiotic] (µg/mL)",
    ylabel="optical density (OD₆₂₀)",
    xscale=log10,
    aspect=AxisAspect(1.25),
    xlabelsize=12,
    ylabelsize=12,
    xticklabelsize=12,
    yticklabelsize=12,
    halign=:left,
)

# Define colors for plot
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1, length(df_raw_group)))

# Loop through days
for (i, d) in enumerate(df_raw_group)
    # Sort data by concentration
    DF.sort!(d, :concentration_ugmL)
    # Plot scatter line
    scatterlines!(
        ax,
        d.concentration_ugmL,
        d.OD,
        color=colors[i],
        label="$(first(d.day))",
        markersize=6,
    )
end # for

# Add colorbar to indicate day
Colorbar(
    gl05B[1, 2],
    colormap=to_colormap(colors),
    limits=(0, length(df_raw_group)),
    label="day",
    ticksvisible=false,
    ticks=0:9:length(df_raw_group),
    halign=:left,
    alignmode=Mixed(; right=0),
    labelsize=12,
    ticklabelsize=12,
    tellheight=true,
    tellwidth=true,
)

# Reduce colgap
# colgap!(gl05B, -35)

# ------------------------------------------------------------------------------
# Fig05C
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
        gl05C[row, col],
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
        # Add error bars
        errorbars!(
            strain.day,
            strain.logic50_mean,
            strain.logic50_mean - strain.logic50_ci_lower,
            strain.logic50_ci_upper - strain.logic50_mean,
            color=colors[strain_num],
        )
    end # for
end # for

# Add global axis labels
Label(
    gl05C[end, :, Bottom()],
    "day",
    fontsize=14,
    padding=(0, 0, 0, 20),
)
Label(
    gl05C[:, 1, Left()],
    "log(IC₅₀)",
    fontsize=14,
    rotation=π / 2,
    padding=(0, 10, 0, 0),
)

# Adjust gap between rows and columns
rowgap!(gl05C, 5)
colgap!(gl05C, 5)

# ------------------------------------------------------------------------------
# Fig05D
# ------------------------------------------------------------------------------

# Add axis for PCA
ax_pca = Axis(
    gl05D[1, 1],
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

# Add axis for VAE
ax_vae = Axis(
    gl05D[1, 2],
    title="VAE",
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    titlesize=12,
    xlabelsize=12,
    ylabelsize=12,
    xticklabelsize=12,
    yticklabelsize=12,
    aspect=AxisAspect(1),
)

# Add axis for latent space
ax_rhvae = Axis(
    gl05D[1, 3],
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

# Pack axis into dictionary
ax_dict = Dict(
    "PCA" => ax_pca,
    "VAE" => ax_vae,
    "RHVAE" => ax_rhvae,
)

# Add axis for reconstruction error
ax_mse = Axis(
    gl05D[1, 4],
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

# -----------------------------------------------------------------------------

# Group data by :parent and :env
df_latent_group = DF.groupby(df_latent, [:parent, :env])

# Loop through groups
for (i, data_group) in enumerate(df_latent_group)
    # Loop through each latent space
    for (j, latent_space) in enumerate(unique(data_group.latent_space))
        # Extract data
        data = data_group[data_group.latent_space.==latent_space, :]
        # Plot data
        scatter!(
            ax_dict[latent_space],
            data.latent1,
            data.latent2,
            markersize=5,
            color=ColorSchemes.glasbey_hv_n256[i],
        )
    end # for
end # for

# -----------------------------------------------------------------------------

# Plot reconstruction error
scatterlines!(
    ax_mse,
    1:n_env,
    pca_mse,
    label="PCA",
    color=Antibiotic.viz.colors()[:gold],
)

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(ic50_std, rhvae(ic50_std).µ)
mse_vae = Flux.mse(ic50_std, vae(ic50_std).µ)

# Plot reconstruction error as horizontal line
hlines!(
    ax_mse,
    mse_vae,
    color=Antibiotic.viz.colors()[:green],
    label="2D VAE",
    linestyle=:dash,
    linewidth=2,
)

# Plot reconstruction error as horizontal line
hlines!(
    ax_mse,
    mse_rhvae,
    color=Antibiotic.viz.colors()[:red],
    label="2D RHVAE",
    linestyle=:dash,
    linewidth=2,
)

# Add legend
Legend(
    gl05D[1, 4, Bottom()],
    ax_mse,
    orientation=:horizontal,
    framevisible=false,
    labelsize=10,
    patchsize=(10, 0),
    patchlabelgap=1,
    padding=(-20, 0, -35, 0),
    colgap=10,
    tellheight=true,
    tellwidth=true,
)

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

Label(
    gl05A_banner[1, 1, Left()], "(A)",
    fontsize=20,
    padding=(0, 20, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl05B_banner[1, 1, Left()], "(B)",
    fontsize=20,
    padding=(0, 20, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl05C_banner[1, 1, Left()], "(C)",
    fontsize=20,
    padding=(0, 20, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl05D_banner[1, 1, Left()], "(D)",
    fontsize=20,
    padding=(0, 20, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Save figure
save("$(fig_dir)/fig06_v02.pdf", fig)

fig
