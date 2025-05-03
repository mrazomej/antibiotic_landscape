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

# Import library for dynamic time warping
import DynamicAxisWarping as DAW

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
rhvae_state_dir = "$(vae_dir)/model_state"
vae_state_dir = "$(vae_dir)/vae_model_state"
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

# Define VAE hyper-parameters in a dictionary
vae_kwargs = (β=Float32(0.1),)

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List RHVAE epoch parameters
rhvae_model_states = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))
# List VAE epoch parameters
vae_model_states = sort(Glob.glob("$(vae_state_dir)/*.jld2"[2:end], "/"))

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(rhvae_model_states[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

# Load VAE model
vae = JLD2.load("$(vae_dir)/vae_model.jld2")["model"]
# Load latest model state
Flux.loadmodel!(vae, JLD2.load(vae_model_states[end])["model_state"])

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

println("Map data to RHVAE latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_rhvae_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Map data to VAE latent space...")

# Map data to latent space
dd_vae_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> vae.encoder(slice).μ,
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
fig = Figure(size=(900, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA space",
)

# Add axis
ax_vae = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="VAE latent space",
)

# Add axis
ax_rhvae = Axis(
    fig[1, 3],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE latent space",
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
    ax_rhvae,
    vec(dd_rhvae_latent[latent=DD.At(:latent1)]),
    vec(dd_rhvae_latent[latent=DD.At(:latent2)]),
    markersize=5,
)

# Plot latent space
scatter!(
    ax_vae,
    vec(dd_vae_latent[latent=DD.At(:latent1)]),
    vec(dd_vae_latent[latent=DD.At(:latent2)]),
    markersize=5,
)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by lineage...")

# Initialize figure
fig = Figure(size=(900, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA space",
)

# Add axis
ax_vae = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="VAE latent space",
)

# Add axis
ax_rhvae = Axis(
    fig[1, 3],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE latent space",
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_rhvae_latent, :lineage))
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
        ax_rhvae,
        vec(dd_rhvae_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_rhvae_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Plot latent space
    scatter!(
        ax_vae,
        vec(dd_vae_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_vae_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space_lineage.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space_lineage.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by time...")

# Initialize figure
fig = Figure(size=(900, 300))

# Add axis
ax_pca = Axis(
    fig[1, 1],
    xlabel="PC1",
    ylabel="PC2",
    aspect=AxisAspect(1),
    title="PCA space",
)

# Add axis
ax_vae = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="VAE latent space",
)

# Add axis
ax_rhvae = Axis(
    fig[1, 3],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="RHVAE latent space",
)

# Define color palette
colors = get(
    ColorSchemes.viridis,
    range(0.0, 1.0, length=length(DD.dims(dd_rhvae_latent, :time)))
)

# Loop over groups
for (i, time) in enumerate(DD.dims(dd_rhvae_latent, :time))
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
        ax_rhvae,
        vec(dd_rhvae_latent[latent=DD.At(:latent1), time=DD.At(time)]),
        vec(dd_rhvae_latent[latent=DD.At(:latent2), time=DD.At(time)]),
        label="$(time)",
        markersize=5,
        color=(colors[i], 0.25),
    )
    # Plot latent space
    scatter!(
        ax_vae,
        vec(dd_vae_latent[latent=DD.At(:latent1), time=DD.At(time)]),
        vec(dd_vae_latent[latent=DD.At(:latent2), time=DD.At(time)]),
        label="$(time)",
        markersize=5,
        color=(colors[i], 0.25),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space_time.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_latent_space_time.png", fig)

fig

## =============================================================================

println("Plotting PCA, VAE, RHVAE, and phenotype space by lineage...")

# Initialize figure
fig = Figure(size=(600, 600))

# Add axis
ax_phenotype = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    title="Original phenotype space",
    aspect=AxisAspect(1)
)

# Add axis
ax_pca = Axis(
    fig[1, 2],
    xlabel="PC1",
    ylabel="PC2",
    title="PCA space",
    aspect=AxisAspect(1)
)

# Add axis
ax_vae = Axis(
    fig[2, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="VAE latent space",
    aspect=AxisAspect(1)
)

# Add axis
ax_rhvae = Axis(
    fig[2, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="RHVAE latent space",
    aspect=AxisAspect(1)
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_rhvae_latent, :lineage))
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
        ax_vae,
        vec(dd_vae_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_vae_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Plot latent space
    scatter!(
        ax_rhvae,
        vec(dd_rhvae_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_rhvae_latent[latent=DD.At(:latent2), lineage=lin]),
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
save("$(fig_dir)/pca_vs_vae_vs_rhvae_phenotype_space_lineage.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_phenotype_space_lineage.png", fig)

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

println("Computing MSE for RHVAE and VAE...")

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(rhvae(fit_mat_std).μ, fit_mat_std)

# Compute MSE for VAE
mse_vae = Flux.mse(vae(fit_mat_std).μ, fit_mat_std)

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
scatterlines!(
    ax_linear,
    n_pcs,
    mse_pca,
    label="PCA",
    color=Antibiotic.viz.colors()[:dark_gold],
    linewidth=2,
)
# Plot PCA MSE
scatterlines!(
    ax_log,
    n_pcs,
    mse_pca,
    label="PCA",
    color=Antibiotic.viz.colors()[:dark_gold],
    linewidth=2,
)

# Plot RHVAE MSE as horizontal line
hlines!.(
    [ax_linear, ax_log],
    mse_rhvae,
    label="2D RHVAE",
    color=Antibiotic.viz.colors()[:dark_blue],
    linestyle=:dash,
    linewidth=2,
)

# Plot VAE MSE as horizontal line
hlines!.(
    [ax_linear, ax_log],
    mse_vae,
    label="2D VAE",
    color=Antibiotic.viz.colors()[:dark_red],
    linestyle=:dash,
    linewidth=2,
)

axislegend(ax_linear)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_mse.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_mse.png", fig)

fig

## =============================================================================

# Extract phenotype data
dd_phenotype = fitnotype_profiles.phenotype[landscape=DD.At(1)]

# Join phenotype, latent and PCA space data all with the same dimensions
dd_join = DD.DimStack(
    (
    phenotype=dd_phenotype,
    rhvae=permutedims(dd_rhvae_latent, (4, 1, 2, 3, 5)),
    vae=permutedims(dd_vae_latent, (4, 1, 2, 3, 5)),
    pca=permutedims(dd_pca, (4, 1, 2, 3, 5)),
),
)

## =============================================================================

println("Computing Z-score transforms...")

# Extract data matrices and standardize data to mean zero and standard deviation 1
# and standard deviation 1
dt_dict = Dict(
    :phenotype => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.phenotype.data, 2, :),
        dims=2
    ),
    :rhvae => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.rhvae.data, 2, :),
        dims=2
    ),
    :vae => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.vae.data, 2, :),
        dims=2
    ),
    :pca => StatsBase.fit(
        StatsBase.ZScoreTransform,
        reshape(dd_join.pca.data, 2, :),
        dims=2
    ),
)

## =============================================================================

# Compute rotation matrix and scale factor with respect to phenotype space
R_dict = Dict()
scale_dict = Dict()

for method in [:rhvae, :vae, :pca]
    # Run procrustes analysis
    proc_result = Antibiotic.geometry.procrustes(
        StatsBase.transform(dt_dict[method], reshape(dd_join[method].data, 2, :)),
        StatsBase.transform(dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)),
        center=true
    )
    # Store rotation matrix and scale factor
    R_dict[method] = proc_result[2]
    scale_dict[method] = proc_result[3]
end

## =============================================================================

println("Plotting rotated PCA, VAE, RHVAE, and phenotype space by lineage...")

# Initialize figure
fig = Figure(size=(600, 600))

# Add axis
ax_phenotype = Axis(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    title="Original phenotype space",
    aspect=AxisAspect(1)
)

# Add axis
ax_pca = Axis(
    fig[1, 2],
    xlabel="PC1",
    ylabel="PC2",
    title="Rotated PCA space",
    aspect=AxisAspect(1)
)

# Add axis
ax_vae = Axis(
    fig[2, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="Rotated VAE latent space",
    aspect=AxisAspect(1)
)

# Add axis
ax_rhvae = Axis(
    fig[2, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="Rotated RHVAE latent space",
    aspect=AxisAspect(1)
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_rhvae_latent, :lineage))
    # Extract PCA data
    data_pca = StatsBase.transform(
        dt_dict[:pca], reshape(dd_join.pca[lineage=lin].data, 2, :)
    )
    # Rotate PCA data
    data_pca_rot = R_dict[:pca] * data_pca
    # Plot PCA
    scatter!(
        ax_pca,
        eachrow(data_pca_rot)...,
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Extract VAE data
    data_vae = StatsBase.transform(
        dt_dict[:vae], reshape(dd_join.vae[lineage=lin].data, 2, :)
    )
    # Rotate VAE data
    data_vae_rot = R_dict[:vae] * data_vae
    # Plot latent space
    scatter!(
        ax_vae,
        eachrow(data_vae_rot)...,
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Extract RHVAE data
    data_rhvae = StatsBase.transform(
        dt_dict[:rhvae], reshape(dd_join.rhvae[lineage=lin].data, 2, :)
    )
    # Rotate RHVAE data
    data_rhvae_rot = R_dict[:rhvae] * data_rhvae
    # Plot latent space
    scatter!(
        ax_rhvae,
        eachrow(data_rhvae_rot)...,
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
    # Extract phenotype data
    data_phenotype = StatsBase.transform(
        dt_dict[:phenotype], reshape(dd_join.phenotype[lineage=lin].data, 2, :)
    )

    # Plot phenotype
    scatter!(
        ax_phenotype,
        eachrow(data_phenotype)...,
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.5),
    )
end # for 

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_phenotype_space_lineage_rotated.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_phenotype_space_lineage_rotated.png", fig)

fig

## =============================================================================

println("Computing Discrete Frechet distance...")

# Group data by lineage, replicate, and evo
dd_group = DD.groupby(dd_join, DD.dims(dd_join)[3:5])

# Initialize empty dataframe to store Frechet distance
df_frechet = DF.DataFrame()

# Loop over groups
for (i, group) in enumerate(dd_group)
    # Extract phenotypic data
    data_phenotype = StatsBase.transform(
        dt_dict[:phenotype],
        dropdims(group.phenotype.data, dims=(3, 4, 5))
    )
    # Extract rotated PCA data
    data_pca = scale_dict[:pca] * R_dict[:pca] * StatsBase.transform(
                   dt_dict[:pca],
                   dropdims(group.pca.data, dims=(3, 4, 5))
               )
    # Extract rotated VAE data
    data_vae = scale_dict[:vae] * R_dict[:vae] * StatsBase.transform(
                   dt_dict[:vae],
                   dropdims(group.vae.data, dims=(3, 4, 5))
               )
    # Extract rotated RHVAE data
    data_rhvae = scale_dict[:rhvae] * R_dict[:rhvae] * StatsBase.transform(
                     dt_dict[:rhvae],
                     dropdims(group.rhvae.data, dims=(3, 4, 5))
                 )

    # Compute Frechet distance
    frechet_pca = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_pca
    )
    frechet_vae = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_vae
    )
    frechet_rhvae = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_rhvae
    )

    # Append to dataframe
    DF.append!(df_frechet, DF.DataFrame(
        lineage=first(values(DD.dims(group)[3])),
        replicate=first(values(DD.dims(group)[4])),
        evo=first(values(DD.dims(group)[5])),
        frechet_pca=frechet_pca,
        frechet_vae=frechet_vae,
        frechet_rhvae=frechet_rhvae
    ))
end # for

## =============================================================================

# Initialize figure
fig = Figure(size=(400, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="Frechet distance",
    ylabel="ECDF",
    title="Discrete Frechet distance",
    aspect=AxisAspect(1)
)

# Plot ECDF
ecdfplot!(
    ax,
    df_frechet.frechet_pca,
    color=ColorSchemes.seaborn_colorblind[1],
    label="PCA",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet.frechet_vae,
    color=ColorSchemes.seaborn_colorblind[2],
    label="VAE",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet.frechet_rhvae,
    color=ColorSchemes.seaborn_colorblind[3],
    label="RHVAE",
    linewidth=2,
)

axislegend(ax, position=:rb)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_frechet_distance.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_frechet_distance.png", fig)

fig

## =============================================================================

# Group data by lineage, replicate, and evo
dd_group = DD.groupby(dd_join, DD.dims(dd_join)[3:5])

# Initialize empty dataframe to store Frechet distance
df_frechet_individual = DF.DataFrame()

# Loop over groups
for (i, group) in enumerate(dd_group)
    # Extract phenotypic data
    data_phenotype = StatsBase.transform(
        dt_dict[:phenotype],
        dropdims(group.phenotype.data, dims=(3, 4, 5))
    )
    # Extract PCA data
    data_pca = StatsBase.transform(
        dt_dict[:pca],
        dropdims(group.pca.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to PCA data
    data_pca_rot, _, _ = Antibiotic.geometry.procrustes(
        data_pca, data_phenotype, center=true
    )
    # Extract VAE data
    data_vae = StatsBase.transform(
        dt_dict[:vae],
        dropdims(group.vae.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to VAE data
    data_vae_rot, _, _ = Antibiotic.geometry.procrustes(
        data_vae, data_phenotype, center=true
    )
    # Extract RHVAE data
    data_rhvae = StatsBase.transform(
        dt_dict[:rhvae],
        dropdims(group.rhvae.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to RHVAE data
    data_rhvae_rot, _, _ = Antibiotic.geometry.procrustes(
        data_rhvae, data_phenotype, center=true
    )

    # Compute Frechet distance
    frechet_pca = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_pca_rot
    )
    frechet_vae = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_vae_rot
    )
    frechet_rhvae = Antibiotic.geometry.discrete_frechet_distance(
        data_phenotype, data_rhvae_rot
    )

    # Append to dataframe
    DF.append!(df_frechet_individual, DF.DataFrame(
        lineage=first(values(DD.dims(group)[3])),
        replicate=first(values(DD.dims(group)[4])),
        evo=first(values(DD.dims(group)[5])),
        frechet_pca=frechet_pca,
        frechet_vae=frechet_vae,
        frechet_rhvae=frechet_rhvae
    ))
end # for

## =============================================================================

# Initialize figure
fig = Figure(size=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="Frechet distance",
    ylabel="ECDF",
    title="Discrete Frechet distance",
    aspect=AxisAspect(1)
)

# Plot ECDF
ecdfplot!(
    ax,
    df_frechet_individual.frechet_pca,
    color=ColorSchemes.seaborn_colorblind[1],
    label="PCA",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet_individual.frechet_vae,
    color=ColorSchemes.seaborn_colorblind[2],
    label="VAE",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet_individual.frechet_rhvae,
    color=ColorSchemes.seaborn_colorblind[3],
    label="RHVAE",
    linewidth=2,
)

axislegend(ax, position=:rb)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_frechet_distance_individual.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_frechet_distance_individual.png", fig)

fig

## =============================================================================

println("Computing Dynamic Time Warping distance with global alignment...")

# Group data by lineage, replicate, and evo
dd_group = DD.groupby(dd_join, DD.dims(dd_join)[3:5])

# Initialize empty dataframe to store Frechet distance
df_dtw = DF.DataFrame()

# Loop over groups
for (i, group) in enumerate(dd_group)
    # Extract phenotypic data
    data_phenotype = StatsBase.transform(
        dt_dict[:phenotype],
        dropdims(group.phenotype.data, dims=(3, 4, 5))
    )
    # Extract rotated PCA data
    data_pca = scale_dict[:pca] * R_dict[:pca] * StatsBase.transform(
                   dt_dict[:pca],
                   dropdims(group.pca.data, dims=(3, 4, 5))
               )
    # Extract rotated VAE data
    data_vae = scale_dict[:vae] * R_dict[:vae] * StatsBase.transform(
                   dt_dict[:vae],
                   dropdims(group.vae.data, dims=(3, 4, 5))
               )
    # Extract rotated RHVAE data
    data_rhvae = scale_dict[:rhvae] * R_dict[:rhvae] * StatsBase.transform(
                     dt_dict[:rhvae],
                     dropdims(group.rhvae.data, dims=(3, 4, 5))
                 )

    # Compute Dynamic Time Warping distance
    dtw_pca, _, _ = DAW.dtw(data_phenotype, data_pca)
    dtw_vae, _, _ = DAW.dtw(data_phenotype, data_vae)
    dtw_rhvae, _, _ = DAW.dtw(data_phenotype, data_rhvae)

    # Append to dataframe
    DF.append!(df_dtw, DF.DataFrame(
        lineage=first(values(DD.dims(group)[3])),
        replicate=first(values(DD.dims(group)[4])),
        evo=first(values(DD.dims(group)[5])),
        dtw_pca=dtw_pca,
        dtw_vae=dtw_vae,
        dtw_rhvae=dtw_rhvae
    ))
end # for

## =============================================================================

# Initialize figure
fig = Figure(size=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="Dynamic Time Warping distance",
    ylabel="ECDF",
    title="Dynamic Time Warping distance",
    aspect=AxisAspect(1)
)

# Plot ECDF
ecdfplot!(
    ax,
    df_dtw.dtw_pca,
    color=ColorSchemes.seaborn_colorblind[1],
    label="PCA",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_dtw.dtw_vae,
    color=ColorSchemes.seaborn_colorblind[2],
    label="VAE",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_dtw.dtw_rhvae,
    color=ColorSchemes.seaborn_colorblind[3],
    label="RHVAE",
    linewidth=2,
)

axislegend(ax, position=:rb)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_dtw_distance_global.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_dtw_distance_global.png", fig)

fig

## =============================================================================

println("Computing Dynamic Time Warping distance with local alignment...")

# Group data by lineage, replicate, and evo
dd_group = DD.groupby(dd_join, DD.dims(dd_join)[3:5])

# Initialize empty dataframe to store Frechet distance
df_dtw_local = DF.DataFrame()

# Loop over groups
for (i, group) in enumerate(dd_group)
    # Extract phenotypic data
    data_phenotype = StatsBase.transform(
        dt_dict[:phenotype],
        dropdims(group.phenotype.data, dims=(3, 4, 5))
    )
    # Extract rotated PCA data
    data_pca = StatsBase.transform(
        dt_dict[:pca],
        dropdims(group.pca.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to PCA data
    data_pca_rot, _, _ = Antibiotic.geometry.procrustes(
        data_pca, data_phenotype, center=true
    )
    # Extract rotated VAE data
    data_vae = StatsBase.transform(
        dt_dict[:vae],
        dropdims(group.vae.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to VAE data
    data_vae_rot, _, _ = Antibiotic.geometry.procrustes(
        data_vae, data_phenotype, center=true
    )
    # Extract rotated RHVAE data
    data_rhvae = StatsBase.transform(
        dt_dict[:rhvae],
        dropdims(group.rhvae.data, dims=(3, 4, 5))
    )
    # Apply procrustes rotation to RHVAE data
    data_rhvae_rot, _, _ = Antibiotic.geometry.procrustes(
        data_rhvae, data_phenotype, center=true
    )

    # Compute Dynamic Time Warping distance
    dtw_pca, _, _ = DAW.dtw(data_phenotype, data_pca_rot)
    dtw_vae, _, _ = DAW.dtw(data_phenotype, data_vae_rot)
    dtw_rhvae, _, _ = DAW.dtw(data_phenotype, data_rhvae_rot)

    # Append to dataframe
    DF.append!(df_dtw_local, DF.DataFrame(
        lineage=first(values(DD.dims(group)[3])),
        replicate=first(values(DD.dims(group)[4])),
        evo=first(values(DD.dims(group)[5])),
        dtw_pca=dtw_pca,
        dtw_vae=dtw_vae,
        dtw_rhvae=dtw_rhvae
    ))
end # for

## =============================================================================

# Initialize figure
fig = Figure(size=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="Dynamic Time Warping distance",
    ylabel="ECDF",
    title="Dynamic Time Warping distance",
    aspect=AxisAspect(1)
)

# Plot ECDF
ecdfplot!(
    ax,
    df_dtw_local.dtw_pca,
    color=ColorSchemes.seaborn_colorblind[1],
    label="PCA",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_dtw_local.dtw_vae,
    color=ColorSchemes.seaborn_colorblind[2],
    label="VAE",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_dtw_local.dtw_rhvae,
    color=ColorSchemes.seaborn_colorblind[3],
    label="RHVAE",
    linewidth=2,
)

axislegend(ax, position=:rb)

# Save figure
save("$(fig_dir)/pca_vs_vae_vs_rhvae_dtw_distance_local.pdf", fig)
save("$(fig_dir)/pca_vs_vae_vs_rhvae_dtw_distance_local.png", fig)

fig