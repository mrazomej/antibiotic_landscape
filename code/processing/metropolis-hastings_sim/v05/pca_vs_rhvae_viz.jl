## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

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

# Import basic math
import LinearAlgebra
import MultivariateStats as MStats
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

println("Loading model...")

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = (K=K, ϵ=ϵ, βₒ=βₒ,)

## =============================================================================

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

# Standardize entire matrix
fit_mat_std = StatsBase.transform(dt, fit_mat)

## =============================================================================

println("Map data to latent space...")

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

## =============================================================================

println("Performing PCA on the data...")

# Perform PCA on the data 
fit_pca = MStats.fit(MStats.PCA, fit_mat_std, maxoutdim=2)

# Define latent space dimensions
pca_dims = DD.Dim{:pca}([:pc1, :pc2])

# Map data to latent space
dd_pca = DD.DimArray(
    dropdims(
        mapslices(slice -> MStats.predict(fit_pca, slice),
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., pca_dims, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Plotting latent space coordinates...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1)
)

# Add axis
ax_latent = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Plot PCA
scatter!(
    ax_pca,
    vec(dd_pca[pca=DD.At(:pc1)]),
    vec(dd_pca[pca=DD.At(:pc2)]),
    markersize=5,
)

# Plot latent space
scatter!(
    ax_latent,
    vec(dd_latent[latent=DD.At(:latent1)]),
    vec(dd_latent[latent=DD.At(:latent2)]),
    markersize=5,
)

# Save figure
save("$(fig_dir)/pca_vs_rhvae_latent_space.pdf", fig)
save("$(fig_dir)/pca_vs_rhvae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by lineage...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1)
)

# Add axis
ax_latent = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_latent, :lineage))
    # Plot PCA
    scatter!(
        ax_pca,
        vec(dd_pca[pca=DD.At(:pc1), lineage=lin]),
        vec(dd_pca[pca=DD.At(:pc2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Plot latent space
    scatter!(
        ax_latent,
        vec(dd_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_rhvae_latent_space_lineage.pdf", fig)
save("$(fig_dir)/pca_vs_rhvae_latent_space_lineage.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by time...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1)
)

# Add axis
ax_latent = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Define color palette
colors = get(
    ColorSchemes.viridis,
    range(0.0, 1.0, length=length(DD.dims(dd_latent, :time)))
)

# Loop over groups
for (i, time) in enumerate(DD.dims(dd_latent, :time))
    # Plot PCA
    scatter!(
        ax_pca,
        vec(dd_pca[pca=DD.At(:pc1), time=DD.At(time)]),
        vec(dd_pca[pca=DD.At(:pc2), time=DD.At(time)]),
        label="$(time)",
        markersize=5,
        color=(colors[i], 0.25),
    )
    # Plot latent space
    scatter!(
        ax_latent,
        vec(dd_latent[latent=DD.At(:latent1), time=DD.At(time)]),
        vec(dd_latent[latent=DD.At(:latent2), time=DD.At(time)]),
        label="$(time)",
        markersize=5,
        color=(colors[i], 0.25),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_rhvae_latent_space_time.pdf", fig)
save("$(fig_dir)/pca_vs_rhvae_latent_space_time.png", fig)

fig

## =============================================================================

println("Plotting PCA, RHVAE, and phenotype space by lineage...")

# Initialize figure
fig = Figure(size=(825, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    title="PCA space",
    aspect=AxisAspect(1)
)

# Add axis
ax_latent = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="RHVAE latent space",
    aspect=AxisAspect(1)
)

# Add axis
ax_phenotype = Axis(
    fig[1, 3],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    title="Original phenotype space",
    aspect=AxisAspect(1)
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_latent, :lineage))
    # Plot PCA
    scatter!(
        ax_pca,
        vec(dd_pca[pca=DD.At(:pc1), lineage=lin]),
        vec(dd_pca[pca=DD.At(:pc2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Plot latent space
    scatter!(
        ax_latent,
        vec(dd_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Plot phenotype
    scatter!(
        ax_phenotype,
        vec(
            fitnotype_profiles.phenotype[
                phenotype=DD.At(:x1), lineage=lin, landscape=1]
        ),
        vec(
            fitnotype_profiles.phenotype[
                phenotype=DD.At(:x2), lineage=lin, landscape=1]
        ),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_rhvae_phenotype_space_lineage.pdf", fig)
save("$(fig_dir)/pca_vs_rhvae_phenotype_space_lineage.png", fig)

fig

## =============================================================================

println("Performing PCA on the data with different number of PCs using SVD...")

# Define number of PCs
n_pcs = 1:15

# Perform SVD on the data
U, S, V = LinearAlgebra.svd(fit_mat_std)

# Initialize vector to store MSE
mse_pca = Vector{Float64}(undef, length(n_pcs))

# Loop over number of PCs
for (i, n_pc) in enumerate(n_pcs)
    # Project data onto first n_pc principal components
    data_proj = U[:, 1:n_pc] * LinearAlgebra.Diagonal(S[1:n_pc]) * V[:, 1:n_pc]'

    # Compute MSE for reconstruction
    mse_pca[i] = Flux.mse(data_proj, fit_mat_std)
end # for

## =============================================================================

println("Computing MSE for RHVAE...")

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(rhvae(fit_mat_std).μ, fit_mat_std)

## =============================================================================

println("Plotting MSE...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax_linear = Axis(
    fig[1, 1],
    xlabel="number of principal components",
    ylabel="mean squared error",
    # yscale=log10,
)

# Add axis
ax_log = Axis(
    fig[1, 2],
    xlabel="number of principal components",
    ylabel="mean squared error",
    yscale=log10,
)
# Plot PCA MSE
scatterlines!(ax_linear, n_pcs, mse_pca, label="PCA",)
# Plot PCA MSE
scatterlines!(ax_log, n_pcs, mse_pca, label="PCA",)

# Plot RHVAE MSE as horizontal line
hlines!.(
    [ax_linear, ax_log],
    mse_rhvae,
    label="2D RHVAE",
    color=:black,
    linestyle=:dash,
)

axislegend(ax_linear)

# Save figure
save("$(fig_dir)/pca_vs_rhvae_mse.pdf", fig)
save("$(fig_dir)/pca_vs_rhvae_mse.png", fig)

fig

## =============================================================================

# Extract phenotype data
dd_phenotype = fitnotype_profiles.phenotype[landscape=DD.At(1)]

# Join phenotype, latent and PCA space data
dd_join = DD.DimStack(
    (phenotype=dd_phenotype, latent=dd_latent, pca=dd_pca),
)
