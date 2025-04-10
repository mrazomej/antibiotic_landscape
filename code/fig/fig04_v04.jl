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
import Distances

# Load Plotting packages
using CairoMakie
using Makie
using Makie.StructArrays
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

# Compute rotation matrix with respect to phenotype space
R_dict = Dict(
    :rhvae => Antibiotic.geometry.procrustes(
        StatsBase.transform(dt_dict[:rhvae], reshape(dd_join.rhvae.data, 2, :)),
        StatsBase.transform(dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)),
        center=false
    )[2],
    :vae => Antibiotic.geometry.procrustes(
        StatsBase.transform(dt_dict[:vae], reshape(dd_join.vae.data, 2, :)),
        StatsBase.transform(dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)),
        center=false
    )[2],
    :pca => Antibiotic.geometry.procrustes(
        StatsBase.transform(dt_dict[:pca], reshape(dd_join.pca.data, 2, :)),
        StatsBase.transform(dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)),
        center=false
    )[2],
)

## =============================================================================

println("Computing MSE for PCA, RHVAE and VAE...")

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

# ------------------------------------------------------------------------------

# Compute MSE for RHVAE
mse_rhvae = Flux.mse(rhvae(fit_mat_std).μ, fit_mat_std)

# Compute MSE for VAE
mse_vae = Flux.mse(vae(fit_mat_std).μ, fit_mat_std)

## =============================================================================

println("Computing Riemannian metric for latent space...")

# Define number of points per axis
n_points = 200

# Extract latent space ranges
latent1_range = range(
    minimum(dd_join.rhvae[latent=DD.At(:latent1)]) - 6,
    maximum(dd_join.rhvae[latent=DD.At(:latent1)]) + 6,
    length=n_points
)
latent2_range = range(
    minimum(dd_join.rhvae[latent=DD.At(:latent2)]) - 6,
    maximum(dd_join.rhvae[latent=DD.At(:latent2)]) + 6,
    length=n_points
)

# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

# Create rotated grid for visualization
# First, create meshgrid of coordinates
x_grid = repeat(latent1_range, 1, n_points)
y_grid = repeat(latent2_range', n_points, 1)

# Apply the same rotation as used for the latent space points
# We need to standardize the grid points first
grid_points = hcat(vec(x_grid), vec(y_grid))'
grid_points_std = StatsBase.transform(dt_dict[:rhvae], grid_points)
grid_points_rot = R_dict[:rhvae] * grid_points_std

# Reshape back to grid format
x_grid_rot = reshape(grid_points_rot[1, :], n_points, n_points)
y_grid_rot = reshape(grid_points_rot[2, :], n_points, n_points)

## =============================================================================

println("Computing Euclidean distances with respect to phenotype space...")

# Compute all pairwise distances
dist_dict = Dict(
    :phenotype => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(
            dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)
        )
    ),
    :rhvae => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:rhvae], reshape(dd_join.rhvae.data, 2, :))
    ),
    :vae => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:vae], reshape(dd_join.vae.data, 2, :))
    ),
    :pca => Distances.pairwise(
        Distances.Euclidean(),
        StatsBase.transform(dt_dict[:pca], reshape(dd_join.pca.data, 2, :))
    ),
)

## =============================================================================
# Static Fig04
## =============================================================================

# Set random seed
Random.seed!(42)

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(700, 700))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add
# Add grid layout for fig04A section banner
gl04A_banner = gl[1, 1:2] = GridLayout()
# Add grid layout for Fig04A
gl04A = gl[2, 1:2] = GridLayout()

# Add grid layout for fig04B section banner
gl04B_banner = gl[3, 1:2] = GridLayout()
# Add grid layout for Fig04B
gl04B = gl[4, 1:2] = GridLayout()

# Add grid layout for fig04C section banner
gl04C_banner = gl[5, 1] = GridLayout()
# Add grid layout for Fig04C
gl04C = gl[6, 1] = GridLayout()

# Add grid layout for fig04B section banner
gl04D_banner = gl[5, 2] = GridLayout()
# Add grid layout for Fig04B
gl04D = gl[6, 2] = GridLayout()

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

println("Adding section banners...")

# Add box for section title
Box(
    gl04A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-35, right=0), # Moves box to the left and right
)

# Add section title
Label(
    gl04A_banner[1, 1],
    "comparison of ground truth and learned latent space coordinates",
    fontsize=12,
    padding=(-25, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl04B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-35, right=0), # Moves box to the left and right
)

# Add section title
Label(
    gl04B_banner[1, 1],
    "pairwise Euclidean distances comparison",
    fontsize=12,
    padding=(-25, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl04C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-35, right=0) # Moves box to the left and right
)

# Add section title
Label(
    gl04C_banner[1, 1],
    "reconstruction error for different models",
    fontsize=12,
    padding=(-25, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl04D_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=0), # Moves box to the left and right
)

# Add section title
Label(
    gl04D_banner[1, 1],
    "RHVAE latent space with Riemannian metric",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=5), # Moves text to the left
    justification=:left,
)

# ------------------------------------------------------------------------------
# Plot Fig04A
# ------------------------------------------------------------------------------

println("Plotting Fig04A...")

# Add axis
ax_phenotype = Axis(
    gl04A[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    title="ground truth\nphenotype space",
    aspect=AxisAspect(1),
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridwidth=0.0,
    ygridwidth=0.0,
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
)

# Add axis
ax_pca = Axis(
    gl04A[1, 2],
    xlabel="PC1",
    ylabel="PC2",
    title="PCA space",
    aspect=AxisAspect(1),
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridwidth=0.0,
    ygridwidth=0.0,
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
)

# Add axis
ax_vae = Axis(
    gl04A[1, 3],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="VAE latent space",
    aspect=AxisAspect(1),
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridwidth=0.0,
    ygridwidth=0.0,
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
)

# Add axis
ax_rhvae = Axis(
    gl04A[1, 4],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="RHVAE latent space",
    aspect=AxisAspect(1),
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridwidth=0.0,
    ygridwidth=0.0,
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
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
        rasterize=true,
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
        rasterize=true,
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
        rasterize=true,
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
        rasterize=true,
    )
end # for 

# ------------------------------------------------------------------------------

# Extract PCA data with metadata
pca_data = dd_join.pca

# Initialize variables to store the maximum points and their metadata
max_left_point = nothing
max_right_point = nothing
max_left_value = -Inf
max_right_value = -Inf

# Iterate over each lineage and time point
for lineage in DD.dims(pca_data, :lineage)
    for time in DD.dims(pca_data, :time)
        for replicate in DD.dims(pca_data, :replicate)
            for evo in DD.dims(pca_data, :evo)
                # Extract the PCA coordinates
                pca_coords = pca_data[
                    lineage=DD.At(lineage),
                    time=DD.At(time),
                    replicate=DD.At(replicate),
                    evo=DD.At(evo)
                ].data
                # Rotate PCA coordinates
                pca_coords = R_dict[:pca] * pca_coords

                # Check if the x-coordinate is less than zero and update
                # max_left_point
                if (pca_coords[1] < 0) && (pca_coords[2] > max_left_value)
                    global max_left_value = pca_coords[2]
                    global max_left_point = (lineage, time, replicate, evo)
                end

                # Check if the x-coordinate is greater than zero and update
                # max_right_point
                if (pca_coords[1] > 0) && (pca_coords[2] > max_right_value)
                    global max_right_value = pca_coords[2]
                    global max_right_point = (lineage, time, replicate, evo)
                end
            end
        end
    end
end

# ------------------------------------------------------------------------------

# Function to plot a point on a given axis
function plot_point!(ax, space, metadata, color)
    # Unpack metadata
    lineage, time, replicate, evo = metadata
    # Extract point data
    point_data = dd_join[space][
        lineage=DD.At(lineage),
        time=DD.At(time),
        replicate=DD.At(replicate),
        evo=DD.At(evo)
    ].data
    # Standardize point data
    point_data_std = StatsBase.transform(dt_dict[space], point_data)
    if space ≠ :phenotype
        # Rotate point data
        point_data_std = R_dict[space] * point_data_std
    end
    # Plot point
    scatter!(
        ax,
        point_data_std[1],
        point_data_std[2],
        markersize=9,
        color=:white,
        marker=:diamond,
        strokecolor=color,
        strokewidth=2,
        rasterize=false,
    )
end

# Plot the left and right points in each latent space
plot_point!(ax_pca, :pca, max_left_point, Antibiotic.viz.colors()[:dark_blue])
plot_point!(ax_pca, :pca, max_right_point, Antibiotic.viz.colors()[:dark_red])

plot_point!(ax_vae, :vae, max_left_point, Antibiotic.viz.colors()[:dark_blue])
plot_point!(ax_vae, :vae, max_right_point, Antibiotic.viz.colors()[:dark_red])

plot_point!(ax_rhvae, :rhvae, max_left_point, Antibiotic.viz.colors()[:dark_blue])
plot_point!(ax_rhvae, :rhvae, max_right_point, Antibiotic.viz.colors()[:dark_red])

plot_point!(ax_phenotype, :phenotype, max_left_point, Antibiotic.viz.colors()[:dark_blue])
plot_point!(ax_phenotype, :phenotype, max_right_point, Antibiotic.viz.colors()[:dark_red])

# ------------------------------------------------------------------------------
# Plot Fig04B
# ------------------------------------------------------------------------------

println("Plotting Fig04B...")

# Extract phenotype-space lower triangular matrix
dist_mat = dist_dict[:phenotype]
dist_phenotype = dist_mat[LinearAlgebra.tril(trues(size(dist_mat)), -1)]

cmap = to_colormap(:BuPu_9)
cmap[1] = RGBAf(1, 1, 1, 1) # make sure background is white

# Loop through latent spaces
for (i, space) in enumerate([:pca, :vae, :rhvae])
    # Extract lower triangular matrix
    dist_mat = dist_dict[space]
    dist_space = dist_mat[LinearAlgebra.tril(trues(size(dist_mat)), -1)]
    # Compute R-squared
    r2 = 1 -
         sum((dist_phenotype .- dist_space) .^ 2) /
         sum((dist_phenotype .- StatsBase.mean(dist_phenotype)) .^ 2)
    # Convert points to StructArray
    points = StructArray{Point2f}((dist_phenotype, dist_space))
    # Initialize axis
    ax = Axis(
        gl04B[1, i],
        title="$(uppercase(string(space))) | R² = $(round(r2[1], digits=2))",
        xlabel="phenotype-space distance",
        ylabel="latent-space distance",
        aspect=AxisAspect(1),
        titlesize=14,
        xlabelsize=14,
        ylabelsize=14,
    )
    # Plot distance scatter plot
    datashader!(ax, points, colormap=cmap, async=false)
    # Add diagonal line
    lines!(
        ax,
        [0, maximum(dist_phenotype)],
        [0, maximum(dist_phenotype)],
        color=:black,
        linewidth=2,
        linestyle=:dash,
    )
    # Set limits
    xlims!(ax, (0, 6))
    ylims!(ax, (0, 6))
end

# Set column gap
colgap!(gl04B, 20)

# ------------------------------------------------------------------------------
# Plot Fig04C
# ------------------------------------------------------------------------------

println("Plotting Fig04C...")

# Add axis
ax_mse = Axis(
    gl04C[1, 1],
    xlabel="number of PCs",
    ylabel="mean squared error",
    yscale=log10,
    aspect=AxisAspect(1.5),
)

scatterlines!(
    ax_mse,
    n_pcs,
    mse_pca,
    label="PCA",
    color=Antibiotic.viz.colors()[:gold],
)

# Plot VAE MSE
hlines!(
    ax_mse,
    mse_vae,
    color=Antibiotic.viz.colors()[:green],
    label="2D VAE",
    linestyle=:dash,
    linewidth=2,
)

# Plot RHVAE MSE
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
    gl04C[1, 1, Top()],
    ax_mse,
    orientation=:horizontal,
    framevisible=false,
    labelsize=11,
    patchsize=(15, 0),
    colgap=5,
    tellheight=true,
    tellwidth=true,
)

# ------------------------------------------------------------------------------
# Plot Fig04D
# ------------------------------------------------------------------------------

println("Plotting Fig04D...")

# Add axis
ax1 = Axis(
    gl04D[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="RHVAE latent\nmetric volume",
    titlesize=12,
    xlabelsize=14,
    ylabelsize=14,
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)
ax2 = Axis(
    gl04D[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    title="fitness profiles \nlatent coordinates",
    titlesize=12,
    xlabelsize=14,
    ylabelsize=14,
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot heatmap of log determinant of metric tensor
# Using surface plot for the rotated grid
hm = surface!(
    ax1,
    x_grid_rot,
    y_grid_rot,
    logdetG,
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
    shading=NoShading,
    rasterize=true,
)

surface!(
    ax2,
    x_grid_rot,
    y_grid_rot,
    logdetG,
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
    shading=NoShading,
    rasterize=true,
)

# Convert to Point2f
latent_points = Point2f.(
    vec(dd_join.rhvae[latent=DD.At(:latent1)]),
    vec(dd_join.rhvae[latent=DD.At(:latent2)]),
)

# Apply the same rotation to the latent points for consistency
latent_points_mat = hcat(
    [p[1] for p in latent_points],
    [p[2] for p in latent_points]
)'
latent_points_std = StatsBase.transform(dt_dict[:rhvae], latent_points_mat)
latent_points_rot = R_dict[:rhvae] * latent_points_std
latent_points_rotated = Point2f.(
    latent_points_rot[1, :], latent_points_rot[2, :]
)

# Plot latent space
scatter!(
    ax2,
    latent_points_rotated,
    markersize=4,
    color=(:white, 0.3),
    rasterize=true,
)

# Find axis limits from minimum and maximum of latent points
xlims!.(
    [ax2, ax1],
    minimum(latent_points_rot[1, :]) - 1,
    maximum(latent_points_rot[1, :]) + 1
)
ylims!.(
    [ax2, ax1],
    minimum(latent_points_rot[2, :]) - 1,
    maximum(latent_points_rot[2, :]) + 1
)

# Add colorbar
Colorbar(
    gl04D[1, 3],
    hm,
    label="√log[det(G)]",
    tellwidth=true,
    tellheight=true,
    halign=:left,
)

# Adjust column gaps
colgap!(gl04D, 5)

# Adjust column sizes
colsize!(gl04D, 1, Auto(1))
colsize!(gl04D, 2, Auto(1))
colsize!(gl04D, 3, Auto(1 / 3))

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust column sizes
colsize!(gl, 1, Auto(2 / 5))
colsize!(gl, 2, Auto(3 / 5))

# Adjust row sizes
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 4, Auto(1))
rowsize!(gl, 6, Auto(1))

# Adjust row gap
rowgap!(gl, 0)
rowgap!(gl, 3, 10)
rowgap!(gl, 4, 10)

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

# Add subplot labels
Label(
    gl04A_banner[1, 1, Left()], "(A)",
    fontsize=20,
    padding=(0, 40, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl04B_banner[1, 1, Left()], "(B)",
    fontsize=20,
    padding=(0, 40, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl04C_banner[1, 1, Left()], "(C)",
    fontsize=20,
    padding=(0, 40, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl04D_banner[1, 1, Left()], "(D)",
    fontsize=20,
    padding=(0, 5, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# ------------------------------------------------------------------------------

println("Saving figure...")
# Save figure
save("$(fig_dir)/fig04_v04.pdf", fig)
save("$(fig_dir)/fig04_v04.png", fig)

fig