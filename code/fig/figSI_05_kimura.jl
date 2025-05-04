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
version_dir = "$(git_root())/output/metropolis-kimura_sim/v05"

# Define simulation directory
sim_dir = "$(version_dir)/sim_evo"
# Define VAE directory
vae_dir = "$(version_dir)/vae"
# Define output directory
# Define output directory
rhvae_state_dir = "$(vae_dir)/rhvae_model_state"
vae_state_dir = "$(vae_dir)/vae_model_state"
# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation landscapes...")

# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]

# Load mutational landscape
genetic_density = JLD2.load("$(sim_dir)/sim_evo.jld2")["genetic_density"]

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(vae_dir)/rhvae_model.jld2"[2:end], "/"))
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
fit_data = Float32.(reshape(fit_data, size(fit_data, 1), :))

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        Float32.(log.(fitnotype_profiles.fitness.data)),
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

# Set random seed
Random.seed!(42)

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(500, 600))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig05A section banner
gl_banner = gl[1, 1] = GridLayout()
# Add grid layout for fig05A
gl_plots = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# # Adjust column sizes
# colsize!(gl, 1, Auto(3))
# colsize!(gl, 2, Auto(1.75))

# # Adjust row sizes
# rowsize!(gl, 2, Auto(1.5))
# rowsize!(gl, 4, Auto(1))

# # Adjust col gaps
# colgap!(gl, -5)
# rowgap!(gl, 5)
# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

println("Adding section banners...")

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-5, right=0), # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "comparison of ground truth and learned fitness landscapes",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)


# ------------------------------------------------------------------------------
# Fig05A
# ------------------------------------------------------------------------------

println("Plotting Fig05...")

# Add grid layout for labels
gl_labels = GridLayout(gl_plots[1:4, 1])
# Add fitness landscape grid layout
gl_fitness = GridLayout(gl_plots[1, 2])
# Add latent space grid layout
gl_rhvae = GridLayout(gl_plots[2, 2])
gl_vae = GridLayout(gl_plots[3, 2])
gl_pca = GridLayout(gl_plots[4, 2])

# Turn grid layouts into dictionary
gl_dict = Dict(
    :rhvae => gl_rhvae,
    :vae => gl_vae,
    :pca => gl_pca,
)

# Adjust col sizes
colsize!(gl_plots, 1, Auto(0.075))
colsize!(gl_plots, 2, Auto(1))

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
# Save figure
# ------------------------------------------------------------------------------

println("Saving figure...")

save("$(fig_dir)/figSI_05_kimura.pdf", fig)
save("$(fig_dir)/figSI_05_kimura.png", fig)

fig
