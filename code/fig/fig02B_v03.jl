## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Import packages for storing results
import DimensionalData as DD

# Import JLD2 for loading results
import JLD2

# Import Distributions
import Distributions

# Import Random
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

println("Defining directories...")

# Define simulation directory
sim_dir = "$(git_root())/output/metropolis-hastings_sim/v05/sim_evo"

# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading simulation results...")

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]
# Load fitness landscapes
fitness_landscapes = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitness_landscapes"]
# Load mutational landscape
genetic_density = JLD2.load(
    "$(sim_dir)/sim_evo.jld2"
)["genetic_density"]

## =============================================================================

println("Plotting alternative landscapes adaptive dynamics...")

# ------------------------------------------------------------------------------
# Define figure layout
# ------------------------------------------------------------------------------

# Define number of rows and columns
n_rows = 2
n_cols = 4

# Initialize figure
fig = Figure(size=(150 * n_cols, 350 * n_rows))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add grid layout for fig02B section banner
gl02B_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02B
gl02B = gl[2, 1] = GridLayout()

# Generate dictionary to store grid layouts for each row
gl_row_dict = Dict()
# Generate dictionary to store grid layouts for (two) subrows per row
gl_subrow_dict = Dict()
# Loop over rows
for i in 1:n_rows
    # Add grid layout for current row
    gl_row_dict[i] = gl02B[i, 1] = GridLayout()
    # Add grid layout for current subrow
    gl_subrow_dict[i] = [
        GridLayout(gl_row_dict[i][1, 1]),
        GridLayout(gl_row_dict[i][2, 1])
    ]
end

# ------------------------------------------------------------------------------
# Add banner box and label
# ------------------------------------------------------------------------------

Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-30, right=-20)
)

Label(
    gl02B_banner[1, 1],
    "trajectories in non-evolution fitness functions",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-10)
)

# ------------------------------------------------------------------------------
# Select reference landscape
# ------------------------------------------------------------------------------

println("Plotting example evolutionary trajectories...")

# Define landscape index
evo_idx = 1

# Extract data
phenotype_trajectory = fitnotype_profiles.phenotype[evo=evo_idx]
# Extract fitness trajectory
fitness_trajectory = fitnotype_profiles.fitness[evo=evo_idx]

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)
# Define limits of fitness space
fitness_lims = (
    y=(
        minimum(fitness_trajectory) - 0.1,
        maximum(fitness_trajectory[landscape=1:(n_rows*n_cols)]) + 0.1,
    ),
)

# Define range of phenotypes to evaluate
x = range(phenotype_lims.x..., length=100)
y = range(phenotype_lims.y..., length=100)

# Evaluate genotype-to-phenotype density map
G = mh.genetic_density(x, y, genetic_density)

# ------------------------------------------------------------------------------
# Plot alternative landscapes fitness trajectories
# ------------------------------------------------------------------------------

# Loop over plots
for i in 1:(n_rows*n_cols)
    # Define landscape index
    landscape_idx = i + 1

    # Define row and column for grid layout
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1

    # Add axis for fitness landscape
    ax_landscape = if col == 1
        Axis(
            gl_subrow_dict[row][1][1, col],
            aspect=AxisAspect(1),
            ylabel="phenotype 2",
            title="env. $landscape_idx",
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    else
        Axis(
            gl_subrow_dict[row][1][1, col],
            aspect=AxisAspect(1),
            title="env. $landscape_idx",
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    end # if col

    # Add y-axis label to first column
    ax_time = if col == 1
        # Add axis for time series
        Axis(
            gl_subrow_dict[row][2][1, col],
            aspect=AxisAspect(4 / 3),
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            ylabel="fitness (a.u.)",
        )
    else
        # Add axis for time series
        Axis(
            gl_subrow_dict[row][2][1, col],
            aspect=AxisAspect(4 / 3),
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    end
    # Set axis limits
    ylims!(ax_time, fitness_lims.y...)

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[landscape_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(x, y, fitness_landscape)

    # Plot fitness landscape
    heatmap!(ax_landscape, x, y, F, colormap=Reverse(:grays))
    # Plot fitness landscape contour lines
    contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)
    # Plot genetic density contour lines
    contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

    # Extract data
    p_traj = phenotype_trajectory[landscape=landscape_idx]
    f_traj = fitness_trajectory[landscape=landscape_idx]

    # Initialize counter
    counter = 1

    # Loop over lineages
    for lin in DD.dims(p_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(p_traj, :replicate)
            # Plot phenotype trajectory
            lines!(
                ax_landscape,
                p_traj[
                    phenotype=DD.At(:x1),
                    lineage=lin,
                    replicate=rep,
                ].data,
                p_traj[
                    phenotype=DD.At(:x2),
                    lineage=lin,
                    replicate=rep,
                ].data,
                color=ColorSchemes.glasbey_hv_n256[counter],
                linewidth=1.5
            )
            # Plot fitness trajectory
            lines!(
                ax_time,
                vec(f_traj[lineage=lin, replicate=rep].data),
                color=ColorSchemes.glasbey_hv_n256[counter],
                linewidth=1.5
            )
            # Increment counter
            counter += 1
        end # for rep
    end # for lin

    # Set axis limits
    xlims!(ax_landscape, phenotype_lims.x...)
    ylims!(ax_landscape, phenotype_lims.y...)
end # for i

# ------------------------------------------------------------------------------
# Add axis labels
# ------------------------------------------------------------------------------

# Loop over rows
for i in 1:n_rows
    # Add x-axis label for fitness landscape
    Label(
        gl_subrow_dict[i][1][end, :, Bottom()],
        "phenotype 1",
        fontsize=16,
    )
    # Add x-axis label for time series
    Label(
        gl_subrow_dict[i][2][end, :, Bottom()],
        "time (a.u.)",
        fontsize=16,
    )
end

# ------------------------------------------------------------------------------
# Adjust subplot spacing
# ------------------------------------------------------------------------------

colgap!(gl02B, 3)
rowgap!(gl02B, 3)

# Save figure
save("$(fig_dir)/fig02B.pdf", fig)

fig

## =============================================================================

println("Creating animated alternative landscapes adaptive dynamics...")

# Initialize figure with same dimensions as static plot
fig = Figure(size=(175 * n_cols, 350 * n_rows))

# Add grid layouts (same structure as static plot)
gl = fig[1, 1] = GridLayout()
gl02B_banner = gl[1, 1] = GridLayout()
gl02B = gl[2, 1] = GridLayout()

# Generate dictionaries for layouts (same as static plot)
gl_row_dict = Dict()
gl_subrow_dict = Dict()
for i in 1:n_rows
    gl_row_dict[i] = gl02B[i, 1] = GridLayout()
    gl_subrow_dict[i] = [
        GridLayout(gl_row_dict[i][1, 1]),
        GridLayout(gl_row_dict[i][2, 1])
    ]
end

# Add banner (same as static plot)
Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-30, right=-20)
)

Label(
    gl02B_banner[1, 1],
    "trajectories in non-evolution fitness functions",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-10)
)

# Create dictionaries to store observables and trajectory data
phenotype_points_dict = Dict()
fitness_points_dict = Dict()
phenotype_trajectories_dict = Dict()
fitness_trajectories_dict = Dict()

# ------------------------------------------------------------------------------
# Initialize all axes and static elements
# ------------------------------------------------------------------------------

for i in 1:(n_rows*n_cols)
    # Define landscape index
    landscape_idx = i + 1
    # Define row and column for grid layout
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1

    # Add axis for fitness landscape
    ax_landscape = if col == 1
        Axis(
            gl_subrow_dict[row][1][1, col],
            aspect=AxisAspect(1),
            ylabel="phenotype 2",
            title="env. $landscape_idx",
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    else
        Axis(
            gl_subrow_dict[row][1][1, col],
            aspect=AxisAspect(1),
            title="env. $landscape_idx",
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    end # if col

    # Set axis limits
    xlims!(ax_landscape, phenotype_lims.x...)
    ylims!(ax_landscape, phenotype_lims.y...)


    # Create axes for time series
    ax_time = if col == 1
        Axis(
            gl_subrow_dict[row][2][1, col],
            aspect=AxisAspect(4 / 3),
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            ylabel="fitness (a.u.)",
        )
    else
        Axis(
            gl_subrow_dict[row][2][1, col],
            aspect=AxisAspect(4 / 3),
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    end # if col

    # Set axis limits
    ylims!(ax_time, fitness_lims.y...)
    xlims!(ax_time, 0, length(DD.dims(fitnotype_profiles, :time)))

    # --------------------------------------------------------------------------
    # Plot static elements
    # --------------------------------------------------------------------------

    F = mh.fitness(x, y, fitness_landscapes[landscape_idx])
    heatmap!(ax_landscape, x, y, F, colormap=Reverse(:grays))
    contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)
    contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

    # --------------------------------------------------------------------------
    # Initialize plots
    # --------------------------------------------------------------------------

    # Initialize trajectory storage for this subplot
    phenotype_trajectories_dict[i] = Dict()
    fitness_trajectories_dict[i] = Dict()

    # Extract trajectory data
    p_traj = phenotype_trajectory[landscape=landscape_idx]
    f_traj = fitness_trajectory[landscape=landscape_idx]

    # Initialize counter
    counter = 1
    # Loop over lineages
    for lin in DD.dims(p_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(p_traj, :replicate)
            # Store full phenotype trajectories
            phenotype_trajectories_dict[i][counter] = p_traj[
                lineage=lin,
                replicate=rep
            ].data

            # Store full fitness trajectories
            fitness_trajectories_dict[i][counter] = vec(
                f_traj[lineage=lin, replicate=rep].data
            )

            # Increment counter
            counter += 1
        end # for rep
    end # for lin

    # Create dictionaries to store observables
    phenotype_points_dict[i] = Dict(
        j => Observable(
            Point2f[phenotype_trajectories_dict[i][j][:, 1]]
        ) for j in 1:length(phenotype_trajectories_dict[i])
    )
    fitness_points_dict[i] = Dict(
        j => Observable(
            Point2f[(1, fitness_trajectories_dict[i][j][1])]
        ) for j in 1:length(fitness_trajectories_dict[i])
    )

    # Plot initial points for each trajectory
    for (j, points) in phenotype_points_dict[i]
        lines!(
            ax_landscape,
            phenotype_points_dict[i][j],
            color=ColorSchemes.glasbey_hv_n256[j],
            linewidth=1.5
        )
        lines!(
            ax_time,
            fitness_points_dict[i][j],
            color=ColorSchemes.glasbey_hv_n256[j],
            linewidth=1.5
        )
    end # for j
end # for i

# ------------------------------------------------------------------------------
# Add time labels
# ------------------------------------------------------------------------------

for i in 1:n_rows
    # Add x-axis label for fitness landscape
    Label(
        gl_subrow_dict[i][1][end, :, Bottom()],
        "phenotype 1",
        fontsize=16,
    )
    # Add x-axis label for time series
    Label(
        gl_subrow_dict[i][2][end, :, Bottom()],
        "time (a.u.)",
        fontsize=16,
    )
end

# ------------------------------------------------------------------------------
# Adjust subplot spacing
# ------------------------------------------------------------------------------

colgap!(gl02B, 3)
rowgap!(gl02B, 3)

# ------------------------------------------------------------------------------
# Create animation
# ------------------------------------------------------------------------------

record(
    fig, "$(fig_dir)/fig02B.mp4",
    1:length(DD.dims(fitnotype_profiles, :time));
    framerate=60
) do frame
    # Update all trajectories in all plots
    for i in 1:(n_rows*n_cols)
        for j in 1:length(phenotype_points_dict[i])
            # Update phenotype trajectory
            new_phenotype_point = Point2f(
                phenotype_trajectories_dict[i][j][:, frame]
            )
            phenotype_points_dict[i][j][] = push!(
                phenotype_points_dict[i][j][],
                new_phenotype_point
            )

            # Update fitness trajectory
            new_fitness_point = Point2f(
                [frame, fitness_trajectories_dict[i][j][frame]]
            )
            fitness_points_dict[i][j][] = push!(
                fitness_points_dict[i][j][], new_fitness_point
            )
        end # for j
    end # for i
end # do frame