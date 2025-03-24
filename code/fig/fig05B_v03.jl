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

# Extract data matrices and standardize data to mean zero and standard deviation
# 1 and standard deviation 1
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

println("Plotting Fig05B...")

Random.seed!(1)

# Initialize figure
fig05B = Figure(size=(800, 450))

# Add grid layout for fig05B section banner
gl05B_banner = fig05B[1, 1] = GridLayout()
# Add grid layout for fig05B
gl05B = fig05B[2, 1] = GridLayout()



# ------------------------------------------------------------------------------

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
    "phenotypic vs. latent trajectories for local and global alignments",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
)

# ------------------------------------------------------------------------------
# Define number of rows and columns
n_rows = 3
n_cols = 3

# Add grid layout for labels
gl_left_labels = gl05B[2, 1] = GridLayout()
gl_top_labels = gl05B[1, 2] = GridLayout()
# Add grid layout for subplots
gl_subplots = gl05B[2, 2] = GridLayout()

# Adjust col sizes
colsize!(gl05B, 1, Auto(1 / 8))
colsize!(gl05B, 2, Auto(1))

# Adjust row sizes
rowsize!(gl05B, 1, Auto(1 / 50))
rowsize!(gl05B, 2, Auto(1))

# Adjust row gap
rowgap!(gl05B, 0)

# Adjust col gap
colgap!(gl05B, 5)

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
    # Add box for method labels
    Box(
        gl_left_labels[col, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=0, right=0, top=25, bottom=25),
    )
    # Add label for method
    Label(
        gl_left_labels[col, 1],
        uppercase(string(methods[col])),
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:left,
        tellheight=false,
        alignmode=Mixed(; left=0),
        rotation=π / 2,
    )

    # Add grid layout for local/global labels
    gl_align_labels = gl_top_labels[1, col] = GridLayout()

    # Add box for local alignment labels
    Box(
        gl_align_labels[1, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=0, right=0, top=0, bottom=0),
    )
    # Add box for global alignment labels
    Box(
        gl_align_labels[1, 2],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=0, right=0, top=0, bottom=0),
    )

    # Add label for local alignment
    Label(
        gl_align_labels[1, 1],
        "local",
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:center,
        tellwidth=false,
        alignmode=Mixed(; left=0),
    )

    # Add label for global alignment
    Label(
        gl_align_labels[1, 2],
        "global",
        fontsize=14,
        padding=(0, 0, 0, 0),
        halign=:center,
        tellwidth=false,
        alignmode=Mixed(; left=0),
    )

    # Reduce colgap
    colgap!(gl_align_labels, 2)

    # Select random group
    group = dd_group[rand(1:length(dd_group))]
    # Loop over columns
    for row in 1:n_rows
        # Add GridLayout for pair of subplots
        gl_subplots_pair = gl_subplots[row, col] = GridLayout()

        # Add axis for local alignment
        ax_local = Axis(
            gl_subplots_pair[1, 1],
            aspect=AxisAspect(1),
        )
        # Add axis for global alignment
        ax_global = Axis(
            gl_subplots_pair[1, 2],
            aspect=AxisAspect(1),
        )
        # Reduce colgap
        colgap!(gl_subplots_pair, 2)

        # Extract phenotype data
        data_phenotype = StatsBase.transform(dt_dict[:phenotype],
            dropdims(group.phenotype.data, dims=(3, 4, 5))
        )
        # Center phenotype data locally
        data_phenotype = data_phenotype .- StatsBase.mean(data_phenotype, dims=2)
        # Plot trajectory in phenotype space for local alignment
        scatterlines!.(
            [ax_local, ax_global],
            Ref(Point2f.(eachcol(data_phenotype))),
            color=Antibiotic.viz.colors()[:dark_blue],
            markersize=6,
            marker=:circle,
            label="ground truth",
        )
        # Extract latent data
        data_latent = dropdims(group[methods[row]].data, dims=(3, 4, 5))

        # Locally align latent data via procrustes
        data_latent_local = Antibiotic.geometry.procrustes(
            data_latent,
            data_phenotype,
            center=true
        )[1]

        # Apply global alignment
        data_latent_global = scale_dict[methods[row]] * R_dict[methods[row]] *
                             StatsBase.transform(
                                 dt_dict[methods[row]], data_latent
                             )

        # Plot trajectory in latent space for local alignment
        scatterlines!(
            ax_local,
            eachrow(data_latent_local)...,
            color=color_dict[methods[row]],
            markersize=6,
            marker=:rect,
            label="latent trajectory",
        )
        # Plot trajectory in latent space for global alignment
        scatterlines!(
            ax_global,
            eachrow(data_latent_global)...,
            color=color_dict[methods[row]],
            markersize=6,
            marker=:rect,
            label="latent trajectory",
        )

        # Plot initial and final points for local alignment
        scatter!.(
            [ax_local, ax_global],
            Ref(Point2f(data_phenotype[:, 1])),
            strokecolor=Antibiotic.viz.colors()[:dark_blue],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!.(
            [ax_local, ax_global],
            Ref(Point2f(data_phenotype[:, end])),
            strokecolor=Antibiotic.viz.colors()[:dark_blue],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        scatter!(
            ax_local,
            Point2f(data_latent_local[:, 1]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!(
            ax_local,
            Point2f(data_latent_local[:, end]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        # Plot initial and final points for global alignment
        scatter!(
            ax_global,
            Point2f(data_latent_global[:, 1]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!(
            ax_global,
            Point2f(data_latent_global[:, end]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        scatter!(
            ax_global,
            Point2f(data_latent_global[:, 1]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:diamond,
        )
        scatter!(
            ax_global,
            Point2f(data_latent_global[:, end]),
            strokecolor=color_dict[methods[row]],
            color=:white,
            markersize=8,
            strokewidth=1.5,
            marker=:utriangle,
        )

        # Hide axis labels
        hidedecorations!(ax_local)
        hidedecorations!(ax_global)
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
    gl05B[2, 2, Bottom()],
    [ground_truth, latent_traj, initial_points, final_points],
    ["ground truth", "latent trajectory", "initial points", "final points"],
    orientation=:horizontal,
    nbanks=1,
    framevisible=false,
    labelsize=11,
    patchsize=(15, 0),
    padding=(0, 0, -5, 0),
    colgap=5,
    tellheight=false,
    tellwidth=false,
)

# Adjust gap between rows and columns
rowgap!(gl_subplots, -5)
colgap!(gl_subplots, 15)

rowgap!(gl_labels, -40)

# Save figure
save("$(fig_dir)/fig05B_v03.pdf", fig05B)

fig05B

