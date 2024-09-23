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

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import LinearAlgebra
import StatsBase
import Random
Random.seed!(42)

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
using PDFmerger: append_pdf!
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()
# Find the path prefix where to put figures
path_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"
# Define output directory
out_dir = "$(git_root())/output$(path_prefix)"
# Define model directory
model_dir = "$(git_root())/output$(path_prefix)/model_state"

# Define figure directory
fig_dir = "$(git_root())/fig$(path_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = (K=K, ϵ=ϵ, βₒ=βₒ,)

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(out_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = Glob.glob("$(model_dir)/*.jld2"[2:end], "/")

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

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

## =============================================================================

println("Load model...")

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(df_meta.model_state[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

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
    # Run :logic50_mean_std through RHVAE
    rhvae_output = rhvae(
        data.logic50_mean_std; latent=true, rhvae_kwargs...
    )
    # Extract latent coordinates
    latent = rhvae_output.encoder.µ
    # Compute MSE
    mse = StatsBase.mean(
        (data.logic50_mean_std .- rhvae_output.decoder.µ) .^ 2
    )
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
            :mse => mse,
        )
    )
end # for 
# Map data to latent space

## =============================================================================

println("Loading data matrix for PCA...")

# Load data matrix
fit_std = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_mean_std"]

# Perform PCA
U, S, V = LinearAlgebra.svd(fit_std)

## =============================================================================

println("Project data to increasing PC space and compute MSE...")

# Define number of PCs to project data
n_dims = 1:8

# Initialize array to store MSE
pca_mse = zeros(Float32, length(n_dims))

# Loop through number of PCs
for n_dim in n_dims
    # Project data to n_dim PCs
    fit_pca = U[:, 1:n_dim] * LinearAlgebra.Diagonal(S[1:n_dim]) * V[:, 1:n_dim]'
    # Compute MSE
    pca_mse[n_dim] = StatsBase.mean((fit_std .- fit_pca) .^ 2)
end # for n_dim in n_dims

## =============================================================================

println("Plot PCA MSE as a function of number of PCs...")

# Initialize figure
fig = Figure(size=(300, 250))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="# PCs",
    ylabel="mean squared error",
)

# Plot MSE
scatterlines!(
    ax, n_dims, pca_mse, label="PCA", color=ColorSchemes.seaborn_colorblind[1]
)

# Add horizontal line for RHVAE average MSE
hlines!(
    ax,
    StatsBase.mean(df_latent.mse),
    label="2D RHVAE",
    color=ColorSchemes.seaborn_colorblind[2],
    linewidth=2
)

# Add legend
axislegend(ax, position=:rt)

# Save figure
save("$(fig_dir)/pca_vs_rhvae_mse.pdf", fig)

fig

## =============================================================================

println("Projecting data to 2D PC space...")

# Project data to 2D PCA space
pca_2d = U[:, 1:2]' * fit_std

# Add to dataframe
DF.insertcols!(
    df_latent,
    :pc1 => pca_2d[1, :],
    :pc2 => pca_2d[2, :],
)

## =============================================================================

println("Plotting 2D PCA space...")


# Initialize figure
fig = Figure(resolution=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1)
)

# Plot data
scatter!(
    ax,
    pca_2d[1, :],
    pca_2d[2, :],
    color=(ColorSchemes.seaborn_colorblind[1], 1),
    markersize=5,
)

# Save figure
save("$(fig_dir)/pca_2d.png", fig)

fig

## =============================================================================

println("Plot 2D PCA and RHVAE latent space...")

# Initialize figure
fig = Figure(resolution=(600, 300))

# Add axis for PCA
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA"
)

# Plot PCA data
scatter!(
    ax_pca,
    pca_2d[1, :],
    pca_2d[2, :],
    color=(ColorSchemes.seaborn_colorblind[1], 1),
    markersize=5,
)

# Add axis for RHVAE
ax_rhvae = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE"
)

# Plot RHVAE data
scatter!(
    ax_rhvae,
    df_latent.latent1,
    df_latent.latent2,
    color=(ColorSchemes.seaborn_colorblind[1], 1),
    markersize=5,
)

# Save figure
save("$(fig_dir)/pca_vs_rhvae_latent.png", fig)

fig

## =============================================================================

println("Plotting 2D PCA and RHVAE latent space colored by stress...")

# Group data by :strain_num
df_group = DF.groupby(df_latent, [:meta])

# Initialize figure
fig = Figure(resolution=(900, 300))

# Add axis for PCA
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA"
)

# Add axis for RHVAE
ax_rhvae = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE"
)

# Loop through groups
for (i, data) in enumerate(df_group)
    # Plot PCA
    scatter!(
        ax_pca,
        data.pc1,
        data.pc2,
        color=(ColorSchemes.glasbey_bw_n256[i], 1),
        markersize=5,
    )
    # Plot RHVAE
    scatter!(
        ax_rhvae,
        data.latent1,
        data.latent2,
        color=(ColorSchemes.glasbey_bw_n256[i], 1),
        markersize=5,
    )
end # for (i, data) in enumerate(df_group)

# Add axis
ax = Axis(
    fig[1, 3],
    xlabel="# PCs",
    ylabel="mean squared error",
    aspect=AxisAspect(1),
)

# Plot MSE
scatterlines!(
    ax, n_dims, pca_mse, label="PCA", color=ColorSchemes.seaborn_colorblind[1]
)

# Add horizontal line for RHVAE average MSE
hlines!(
    ax,
    StatsBase.mean(df_latent.mse),
    label="2D RHVAE",
    color=ColorSchemes.seaborn_colorblind[2],
    linewidth=2
)

# Add legend
axislegend(ax, position=:rt)

# Save figure
save("$(fig_dir)/pca_vs_rhvae_latent_stress.png", fig)
save("$(fig_dir)/pca_vs_rhvae_latent_stress.pdf", fig)

fig