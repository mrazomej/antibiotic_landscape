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

# Define version directory
version_dir = "$(git_root())/output/metropolis-hastings_sim/v05"

# Define simulation directory
sim_dir = "$(version_dir)/sim_evo"
# Define VAE directory
vae_dir = "$(version_dir)/vae"
# Define output directory
# Define output directory
rhvae_state_dir = "$(vae_dir)/model_state"
vae_state_dir = "$(vae_dir)/vae_model_state"
# Define output directory
fig_dir = "$(git_root())/fig/main"

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
pca_dims = DD.Dim{:latent}([:latent1, :latent2])

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

println("Joining data into single structure...")
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

println("Plotting Fig05C...")

Random.seed!(1)

# Initialize figure
fig05C = Figure(size=(300, 300))

# Add grid layout for fig05B section banner
gl05C_banner = fig05C[1, 1] = GridLayout()
# Add grid layout for fig05B
gl05C = fig05C[2, 1] = GridLayout()

# Add box for section title
Box(
    gl05C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-55, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05C_banner[1, 1],
    "distance between phenotypic and latent trajectories",
    fontsize=12,
    padding=(-50, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add axis
ax = Axis(
    gl05C[1, 1],
    xlabel="Fréchet distance",
    ylabel="ECDF",
    aspect=AxisAspect(1),
    xticks=(0:2:6)
)

# Plot ECDF
ecdfplot!(
    ax,
    df_frechet.frechet_pca,
    color=color_dict[:pca],
    label="PCA",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet.frechet_vae,
    color=color_dict[:vae],
    label="VAE",
    linewidth=2,
)
ecdfplot!(
    ax,
    df_frechet.frechet_rhvae,
    color=color_dict[:rhvae],
    label="RHVAE",
    linewidth=2,
)

axislegend(ax, position=:rb, framevisible=false)

# Save figure
save("$(fig_dir)/fig05C_v02.pdf", fig05C)

fig05C