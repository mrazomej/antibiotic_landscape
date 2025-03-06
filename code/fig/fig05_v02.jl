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

println("Loading simulation landscapes...")

# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]

# Load mutational landscape
genetic_density = JLD2.load("$(sim_dir)/sim_evo.jld2")["genetic_density"]

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

# Compute rotation matrix and scale factor with respect to phenotype space
R_dict = Dict()
scale_dict = Dict()

for method in [:rhvae, :vae, :pca]
    # Run procrustes analysis
    proc_result = Antibiotic.stats.procrustes(
        StatsBase.transform(dt_dict[method], reshape(dd_join[method].data, 2, :)),
        StatsBase.transform(dt_dict[:phenotype], reshape(dd_join.phenotype.data, 2, :)),
        center=true
    )
    # Store rotation matrix and scale factor
    R_dict[method] = proc_result[2]
    scale_dict[method] = proc_result[3]
end

## =============================================================================

println("Defining latent space ranges...")

# Define number of points per axis
n_points = 200

# Extract latent space ranges for each method
latent_ranges = Dict()
for method in [:pca, :vae, :rhvae]
    latent_ranges[method] = (
        range(
            minimum(dd_join[method][latent=DD.At(:latent1)]) - 5,
            maximum(dd_join[method][latent=DD.At(:latent1)]) + 5,
            length=n_points
        ),
        range(
            minimum(dd_join[method][latent=DD.At(:latent2)]) - 5,
            maximum(dd_join[method][latent=DD.At(:latent2)]) + 5,
            length=n_points
        )
    )
end

# Define latent points to evaluate
z_mat = reduce(
    hcat,
    [[x, y] for x in latent_ranges[:rhvae][1], y in latent_ranges[:rhvae][2]]
)

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

# Create dictionary to store grids for each method
grid_rot = Dict()

# Loop through each method
for method in [:pca, :vae, :rhvae]
    # Create meshgrid of coordinates
    x_grid = repeat(latent_ranges[method][1], 1, n_points)
    y_grid = repeat(latent_ranges[method][2]', n_points, 1)

    # Apply rotation after standardizing grid points
    grid_points = hcat(vec(x_grid), vec(y_grid))'
    grid_points_std = StatsBase.transform(dt_dict[method], grid_points)
    grid_points_rot = R_dict[method] * grid_points_std

    # Store original and rotated grids in dictionary
    grid_rot[method] = (
        x=x_grid,
        y=y_grid,
        x_rot=reshape(grid_points_rot[1, :], n_points, n_points),
        y_rot=reshape(grid_points_rot[2, :], n_points, n_points)
    )
end

# Initialize dictionary to store latent grids for each method
latent_grid_dict = Dict()

# Loop through each method to create latent grids
for method in [:pca, :vae, :rhvae]
    latent_grid_dict[method] = [
        Float32.([x, y]) for (x, y) in IterTools.product(
            latent_ranges[method][1], latent_ranges[method][2]
        )
    ]
end

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.90 .< logdetG .≤ maximum(logdetG))

## =============================================================================

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
pheno1 = range(phenotype_lims.x..., length=n_points)
pheno2 = range(phenotype_lims.y..., length=n_points)

# Create meshgrid for genetic density
G = mh.genetic_density(pheno1, pheno2, genetic_density)

## =============================================================================

# Initialize dictionary to store latent points for each method
latent_points_dict = Dict()

# Loop through each method
for method in [:rhvae, :vae, :pca]
    # Convert to Point2f
    latent_points = Point2f.(
        vec(dd_join[method][latent=DD.At(:latent1)]),
        vec(dd_join[method][latent=DD.At(:latent2)]),
    )

    # Apply the same rotation to the latent points for consistency
    latent_points_mat = hcat(
        [p[1] for p in latent_points],
        [p[2] for p in latent_points]
    )'
    latent_points_std = StatsBase.transform(dt_dict[method], latent_points_mat)
    latent_points_rot = (R_dict[method] * latent_points_std) * scale_dict[method]

    # Store in dictionary
    latent_points_dict[method] = latent_points_rot
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
    frechet_pca = Antibiotic.stats.discrete_frechet_distance(
        data_phenotype, data_pca
    )
    frechet_vae = Antibiotic.stats.discrete_frechet_distance(
        data_phenotype, data_vae
    )
    frechet_rhvae = Antibiotic.stats.discrete_frechet_distance(
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

# Set random seed
Random.seed!(42)

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(800, 650))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig05A section banner
gl05A_banner = gl[1, 1] = GridLayout()
# Add grid layout for fig05A
gl05A = gl[2:4, 1] = GridLayout()

# Add grid layout for fig05B section banner
gl05B_banner = gl[1, 2] = GridLayout()
# Add grid layout for fig05B
gl05B = gl[2, 2] = GridLayout()

# Add grid layout for fig05C section banner
gl05C_banner = gl[3, 2] = GridLayout()
# Add grid layout for fig05C
gl05C = gl[4, 2] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust column sizes
colsize!(gl, 1, Auto(3))
colsize!(gl, 2, Auto(1.75))

# Adjust row sizes
rowsize!(gl, 2, Auto(1.5))
rowsize!(gl, 4, Auto(1))

# Adjust col gaps
colgap!(gl, -5)
rowgap!(gl, 5)
# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

println("Adding section banners...")

# Add box for section title
Box(
    gl05A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=15, right=0), # Moves box to the left and right
)

# Add section title
Label(
    gl05A_banner[1, 1],
    "comparison of ground truth and learned fitness landscapes",
    fontsize=12,
    padding=(20, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# Add box for section title
Box(
    gl05B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05B_banner[1, 1],
    "phenotypic vs. latent trajectories",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)


# Add box for section title
Box(
    gl05C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl05C_banner[1, 1],
    "distance between phenotypic and latent trajectories",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# ------------------------------------------------------------------------------
# Fig05A
# ------------------------------------------------------------------------------

println("Plotting Fig05A...")

# Add grid layout for labels
gl_labels = GridLayout(gl05A[1:4, 1])
# Add fitness landscape grid layout
gl_fitness = GridLayout(gl05A[1, 2])
# Add latent space grid layout
gl_rhvae = GridLayout(gl05A[2, 2])
gl_vae = GridLayout(gl05A[3, 2])
gl_pca = GridLayout(gl05A[4, 2])

# Turn grid layouts into dictionary
gl_dict = Dict(
    :rhvae => gl_rhvae,
    :vae => gl_vae,
    :pca => gl_pca,
)

# Adjust col sizes
colsize!(gl05A, 1, Auto(0.075))
colsize!(gl05A, 2, Auto(1))

# ------------------------------------------------------------------------------

for i in 1:4
    Box(
        gl_labels[i, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=0, right=-5, top=0, bottom=0),
    )
end

# Ground truth fitness landscape
Label(
    gl_labels[1, 1],
    "ground truth\nfitness landscape",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    tellheight=false,
    alignmode=Mixed(; left=0),
    rotation=π / 2,
)
# Add labels for each method
for (i, method) in enumerate([:rhvae, :vae, :pca])
    Label(
        gl_labels[i+1, 1],
        "$(uppercase(string(method))) inferred\nfitness landscape",
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:left,
        tellwidth=false,
        tellheight=false,
        alignmode=Mixed(; left=0),
        rotation=π / 2,
    )
end

# Define environment indexes to use
env_idxs = [1, 8, 25, 36]

# ------------------------------------------------------------------------------
# Plot fitness landscapes
# ------------------------------------------------------------------------------

# Initialize dictionary to store axes
ax_dict = Dict(
    method => Dict{Int,Axis}()
    for method in [:phenotype, :rhvae, :vae, :pca]
)

# Loop over fitness landscapes
for (i, env_idx) in enumerate(env_idxs)
    # Add axis
    ax_dict[:phenotype][env_idx] = Axis(
        gl_fitness[1, i],
        title="env. $(env_idx)",
        titlesize=14,
        aspect=AxisAspect(1),
    )
    ax = ax_dict[:phenotype][env_idx]
    # Remove axis labels
    hidedecorations!(ax)

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[env_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(pheno1, pheno2, fitness_landscape)

    # Plot fitness landscape
    heatmap!(ax, pheno1, pheno2, F, colormap=:algae, rasterize=true)
    # Plot fitness landscape contour lines
    contour!(ax, pheno1, pheno2, F, color=:black, linestyle=:dash)

    # Loop over methods
    for method in [:rhvae, :vae, :pca]
        # Add axis for latent fitness landscape
        ax_dict[method][env_idx] = Axis(gl_dict[method][1, i], aspect=AxisAspect(1))
        ax = ax_dict[method][env_idx]
        # Remove axis labels
        hidedecorations!(ax)

        # Map latent grid to output space
        F_latent = if method == :rhvae
            getindex.(
                getindex.(rhvae.vae.decoder.(latent_grid_dict[method]), :μ),
                env_idx
            )
        elseif method == :vae
            getindex.(
                getindex.(vae.decoder.(latent_grid_dict[method]), :μ),
                env_idx
            )
        else # :pca
            getindex.(
                MStats.reconstruct.(Ref(fit_pca), latent_grid_dict[method]),
                env_idx
            )
        end

        if method == :rhvae
            # Apply mask
            F_latent_masked = (mask .* minimum(F_latent)) .+ (F_latent .* .!mask)
        else
            F_latent_masked = F_latent
        end

        # Plot latent fitness landscape
        surface!(
            ax,
            grid_rot[method].x_rot,
            grid_rot[method].y_rot,
            F_latent_masked,
            colormap=:algae,
            shading=NoShading,
            rasterize=true,
        )

        contour!(
            ax,
            grid_rot[method].x_rot,
            grid_rot[method].y_rot,
            F_latent_masked,
            color=:black,
            linestyle=:dash,
            levels=7,
        )

        # Define pad based on method
        pad = method == :pca ? 0 : 1
        # Find axis limits from minimum and maximum of latent points
        xlims!(
            ax,
            minimum(latent_points_dict[method][1, :]) - pad,
            maximum(latent_points_dict[method][1, :]) + pad
        )
        ylims!(
            ax,
            minimum(latent_points_dict[method][2, :]) - pad,
            maximum(latent_points_dict[method][2, :]) + pad
        )
    end # for method
end # for env_idx

# ------------------------------------------------------------------------------
# Add landscape axis labels
# ------------------------------------------------------------------------------

Label(
    gl_fitness[end, :, Bottom()],
    "phenotype 1",
    fontsize=14,
    padding=(0, 0, 0, 0),
)
Label(
    gl_fitness[:, 1, Left()],
    "phenotype 2",
    fontsize=14,
    rotation=π / 2,
    padding=(0, 0, 0, 0),
)

# Adjust col sizes
colgap!(gl_fitness, 5)

# Add global axis labels for each method
for method in [:pca, :vae, :rhvae]
    Label(
        gl_dict[method][end, :, Bottom()],
        method == :pca ? "PC 1" : "latent dimension 1",
        fontsize=14,
        padding=(0, 0, 0, 0),
    )
    Label(
        gl_dict[method][1, :, Left()],
        method == :pca ? "PC 2" : "latent dimension 2",
        fontsize=14,
        rotation=π / 2,
        padding=(0, 0, 0, 0),
    )

    colgap!(gl_dict[method], 5)
end

# ------------------------------------------------------------------------------
# Fig05B
# ------------------------------------------------------------------------------

println("Plotting Fig05B...")

Random.seed!(1)

# Define number of rows and columns
n_rows = 3
n_cols = 3

# Add grid layout for labels
gl_labels = gl05B[1, 1] = GridLayout()
# Add grid layout for subplots
gl_subplots = gl05B[1, 2] = GridLayout()

# Adjust col sizes
colsize!(gl05B, 1, Auto(1 / 8))
colsize!(gl05B, 2, Auto(1))

# Group data by lineage, replicate, and evo
dd_group = DD.groupby(dd_join, DD.dims(dd_join)[3:5])

# Define methods
methods = [:rhvae, :vae, :pca]

# Define colors for each method
color_dict = Dict(
    :rhvae => Antibiotic.viz.colors()[:red],
    :vae => Antibiotic.viz.colors()[:green],
    :pca => Antibiotic.viz.colors()[:gold],
)

# Loop over rows
for col in 1:n_cols
    # Add box for labels
    Box(
        gl_labels[col, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=0, right=0, top=25, bottom=25),
    )
    # Add label for method
    Label(
        gl_labels[col, 1],
        uppercase(string(methods[col])),
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:left,
        # tellwidth=false,
        tellheight=false,
        alignmode=Mixed(; left=0),
        rotation=π / 2,
    )

    # Select random group
    group = dd_group[rand(1:length(dd_group))]
    # Loop over columns
    for row in 1:n_rows
        # Add axis
        ax = Axis(
            gl_subplots[row, col],
            aspect=AxisAspect(1),
        )
        # Extract phenotype data
        data_phenotype = dropdims(group.phenotype.data, dims=(3, 4, 5))
        # Center phenotype data locally
        data_phenotype = data_phenotype .- StatsBase.mean(data_phenotype, dims=2)
        # Plot trajectory in phenotype space
        scatterlines!(
            ax,
            eachrow(data_phenotype)...,
            color=Antibiotic.viz.colors()[:dark_blue],
            markersize=6,
            marker=:circle,
            label="ground truth",
        )
        # Extract latent data
        data_latent = dropdims(group[methods[row]].data, dims=(3, 4, 5))
        # Align latent data via procrustes
        data_latent = Antibiotic.stats.procrustes(
            data_latent,
            data_phenotype,
            center=true
        )[1]
        # Plot trajectory in latent space
        scatterlines!(
            ax,
            eachrow(data_latent)...,
            color=color_dict[methods[row]],
            markersize=6,
            marker=:rect,
            label="latent trajectory",
        )
        # Plot initial and final points
        scatter!(
            ax,
            Point2f(data_phenotype[:, 1]),
            strokecolor=Antibiotic.viz.colors()[:dark_blue],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!(
            ax,
            Point2f(data_phenotype[:, end]),
            strokecolor=Antibiotic.viz.colors()[:dark_blue],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        scatter!(
            ax,
            Point2f(data_latent[:, 1]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!(
            ax,
            Point2f(data_latent[:, end]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        # Hide axis labels
        hidedecorations!(ax)
    end # for col
end # for row

# Create manual legend elements
ground_truth = [
    LineElement(color=Antibiotic.viz.colors()[:dark_blue]),
    MarkerElement(
        color=Antibiotic.viz.colors()[:dark_blue],
        marker=:circle,
        markersize=6
    )
]
latent_traj = [
    LineElement(color=:gray),
    MarkerElement(
        color=:gray,
        marker=:rect,
        markersize=6
    )
]
initial_points = [
    MarkerElement(
        strokecolor=:gray,
        color=:white,
        marker=:diamond,
        markersize=8,
        strokewidth=1.5,
    )
]
final_points = [
    MarkerElement(
        strokecolor=:gray,
        color=:white,
        marker=:utriangle,
        markersize=8,
        strokewidth=1.5,
    )
]

Legend(
    gl05B[1, 2, Top()],
    [ground_truth, latent_traj, initial_points, final_points],
    ["ground truth", "latent trajectory", "initial points", "final points"],
    orientation=:horizontal,
    nbanks=2,
    framevisible=false,
    labelsize=11,
    patchsize=(15, 0),
    padding=(0, 0, 0, 0),
    colgap=5,
    tellheight=false,
    tellwidth=false,
)

# Adjust gap between rows and columns
rowgap!(gl_subplots, -40)
colgap!(gl_subplots, 2)

rowgap!(gl_labels, -40)

# ------------------------------------------------------------------------------
# Fig05C
# ------------------------------------------------------------------------------

println("Plotting Fig05C...")

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

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

println("Adding subplot labels...")

# Add subplot labels
Label(
    gl05A_banner[1, 1, Left()], "(A)",
    fontsize=20,
    padding=(0, -10, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)
# Add subplot labels
Label(
    gl05B_banner[1, 1, Left()], "(B)",
    fontsize=20,
    padding=(0, 10, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Add subplot labels
Label(
    gl05C_banner[1, 1, Left()], "(C)",
    fontsize=20,
    padding=(0, 10, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# ------------------------------------------------------------------------------
# Save figure
# ------------------------------------------------------------------------------

println("Saving figure...")

save("$(fig_dir)/fig05_v02.pdf", fig)
save("$(fig_dir)/fig05_v02.png", fig)

fig
