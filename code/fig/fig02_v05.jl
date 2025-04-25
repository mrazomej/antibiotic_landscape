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

# Define whether animation should be created
create_animation = false

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
# Static Fig02
## =============================================================================

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(600, 550))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig02A section banner
gl02A_banner = gl[1, 1] = GridLayout()
# Add grid layout for Fig02A
gl02A = gl[2, 1] = GridLayout()

# Add grid layout for fig02B section banner
gl02B_banner = gl[1, 2] = GridLayout()
# Add grid layout for Fig02B
gl02B = gl[2, 2] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust subplot proportions
# ------------------------------------------------------------------------------

# Adjust column sizes
colsize!(gl, 1, Auto(1))
colsize!(gl, 2, Auto(2))

# ------------------------------------------------------------------------------
# Add section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl02A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-12.5, right=-5) # Moves box to the left and right
)

# Add section title
Label(
    gl02A_banner[1, 1],
    "adaptive dynamics in selection\nenvironment",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-10) # Moves text to the left
)

Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=5, right=-10)
)

Label(
    gl02B_banner[1, 1],
    "evaluation of fitness in diverse environments",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=10)
)


# ------------------------------------------------------------------------------
# Fig02A
# ------------------------------------------------------------------------------

# Add grid layout for manual legend
gl02A_legend = gl02A[1, 1] = GridLayout()
# Add grid layout for Fig02A fitness landscape
gl02A_landscape = gl02A[2, 1] = GridLayout()
# Add grid layout for Fig02A time series
gl02A_time = gl02A[3, 1] = GridLayout()

println("Plotting example evolutionary trajectories...")

# Define landscape index
landscape_idx = 1

# Extract data
phenotype_trajectory = fitnotype_profiles.phenotype[
    evo=DD.At(landscape_idx), landscape=DD.At(landscape_idx)
]
# Extract fitness trajectory
fitness_trajectory = fitnotype_profiles.fitness[
    evo=DD.At(landscape_idx), landscape=DD.At(landscape_idx)
]

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
x = range(phenotype_lims.x..., length=100)
y = range(phenotype_lims.y..., length=100)

# Create meshgrid
F = mh.fitness(x, y, fitness_landscapes[landscape_idx])
G = mh.genetic_density(x, y, genetic_density)

# Add axis for fitness landscape
ax_landscape = Axis(
    gl02A_landscape[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    title="env. 1 (selection)",
    titlesize=14,
    aspect=AxisAspect(1),
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot fitness landscape
heatmap!(ax_landscape, x, y, F, colormap=:algae)

# Plot contour lines for fitness landscape
contour!(ax_landscape, x, y, F, color=:black, linestyle=:dash)

# Plot contour lines for genetic density
contour!(ax_landscape, x, y, G, color=:white, linestyle=:solid)

# ------------------------------------------------------------------------------

# Add axis for time series
ax_time = Axis(
    gl02A_time[1, 1],
    xlabel="time",
    ylabel="fitness",
    aspect=AxisAspect(4 / 3),
    yticklabelsvisible=false,
    xticklabelsvisible=false
)

# ------------------------------------------------------------------------------

# Define alpha value for trajectories
alpha = 0.25

# Define number of non-transparent trajectories
n_alpha = 5

# Initialize counter for lineages
counter = 1
# Initialize counter for non-transparent trajectories
n_transp = 1
# Loop over lineages
for lin in DD.dims(phenotype_trajectory, :lineage)
    # Loop over replicates
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Define color and linewidth
        if counter % n_alpha == 0
            color = ColorSchemes.seaborn_colorblind[n_transp]
            linewidth = 1.5
            # Increment counter for non-transparent trajectories
            global n_transp += 1
        else
            color = (ColorSchemes.glasbey_hv_n256[counter], alpha)
            linewidth = 1
        end # if counter
        # Plot trajectory
        lines!(
            ax_landscape,
            eachrow(phenotype_trajectory[
                lineage=DD.At(lin),
                replicate=DD.At(rep),
            ].data)...,
            color=color,
            linewidth=linewidth
        )
        # Plot fitness trajectory
        lines!(
            ax_time,
            vec(fitness_trajectory[
                lineage=DD.At(lin),
                replicate=DD.At(rep),
            ].data),
            color=color,
            linewidth=linewidth
        )
        # Plot initial and final phenotype
        if counter % n_alpha == 0
            point_init = scatter!(
                ax_landscape,
                Point2f(phenotype_trajectory[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                    time=DD.At(0),
                ]),
                color=color,
                markersize=10,
                marker=:xcross,
                strokecolor=:black,
                strokewidth=1.5,
            )
            point_final = scatter!(
                ax_landscape,
                Point2f(phenotype_trajectory[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                    time=DD.At(last(DD.dims(phenotype_trajectory, :time))),
                ]),
                color=color,
                markersize=10,
                marker=:utriangle,
                strokecolor=:black,
                strokewidth=1.5,
            )
            # Translate to front of other trajectories
            translate!(point_init, 0, 0, 100)
            translate!(point_final, 0, 0, 100)
        end # if counter
        # Increment counter
        global counter += 1
    end # for rep
end # for lin

# Set axis limits
xlims!(ax_landscape, phenotype_lims.x...)
ylims!(ax_landscape, phenotype_lims.y...)

# ------------------------------------------------------------------------------

# Add legend manually for landscape
elems = [
    MarkerElement(
        color="#E6E6EF",
        strokecolor=:black,
        marker=:xcross,
        strokewidth=1.5,
        markersize=13,
    ),
    MarkerElement(
        color="#E6E6EF",
        strokecolor=:black,
        marker=:utriangle,
        strokewidth=1.5,
        markersize=13,
    ),
]

Legend(
    gl02A_legend[1, 1],
    elems,
    ["initial phenotype", "evolved phenotype"],
    tellwidth=false,
    tellheight=false,
    framevisible=false,
    labelsize=12,
)

# Adjust subplot spacing
rowgap!(gl02A, 15)

# Adjust row size
rowsize!(gl02A, 1, Auto(1 / 4))
rowsize!(gl02A, 2, Auto(1))
rowsize!(gl02A, 3, Auto(1))

# ------------------------------------------------------------------------------
# Fig02B
# ------------------------------------------------------------------------------

# Define number of rows and columns
n_rows = 2
n_cols = 3

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
    # Adjust row size
    rowsize!(gl_row_dict[i], 1, Auto(1))
    rowsize!(gl_row_dict[i], 2, Auto(3 / 4))
    # Adjust row gap
    rowgap!(gl_row_dict[i], 10)
end

# Adjust subplot spacing
rowgap!(gl02B, 5)

# ------------------------------------------------------------------------------
# Select reference landscape
# ------------------------------------------------------------------------------

println("Plotting example evolutionary trajectories...")

# Define landscape index
evo_idx = 1

# Extract data
phenotype_trajectory = fitnotype_profiles.phenotype[evo=DD.At(evo_idx)]
# Extract fitness trajectory
fitness_trajectory = fitnotype_profiles.fitness[evo=DD.At(evo_idx)]

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)
# Define limits of fitness space
fitness_lims = (
    y=(
        minimum(fitness_trajectory) - 0.5,
        maximum(fitness_trajectory[landscape=1:(n_rows*n_cols)]) + 0.5,
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
            ylabelsize=12,
            title="env. $landscape_idx",
            titlesize=13,
            xticklabelsvisible=false,
            yticklabelsvisible=false,
        )
    else
        Axis(
            gl_subrow_dict[row][1][1, col],
            aspect=AxisAspect(1),
            title="env. $landscape_idx",
            titlesize=13,
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
            ylabel="fitness",
            ylabelsize=12,
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

    # Initialize counter for lineages
    local counter = 1
    # Initialize counter for non-transparent trajectories
    local n_transp = 1

    # Loop over lineages
    for lin in DD.dims(p_traj, :lineage)
        # Loop over replicates
        for rep in DD.dims(p_traj, :replicate)
            # Define color and linewidth
            if counter % n_alpha == 0
                color = ColorSchemes.seaborn_colorblind[n_transp]
                linewidth = 1.5
                # Increment counter for non-transparent trajectories
                n_transp += 1
            else
                color = (ColorSchemes.glasbey_hv_n256[counter], alpha)
                linewidth = 1
            end # if counter
            # Plot phenotype trajectory
            lines!(
                ax_landscape,
                eachrow(p_traj[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                ].data)...,
                color=color,
                linewidth=linewidth
            )
            # Plot fitness trajectory
            lines!(
                ax_time,
                vec(f_traj[
                    lineage=DD.At(lin),
                    replicate=DD.At(rep),
                ].data),
                color=color,
                linewidth=linewidth
            )
            # Plot initial and final phenotype
            if counter % n_alpha == 0
                point_init = scatter!(
                    ax_landscape,
                    Point2f(p_traj[
                        lineage=DD.At(lin),
                        replicate=DD.At(rep),
                        time=DD.At(0),
                    ]),
                    color=color,
                    markersize=8,
                    marker=:xcross,
                    strokecolor=:black,
                    strokewidth=1,
                )
                point_final = scatter!(
                    ax_landscape,
                    Point2f(p_traj[
                        lineage=DD.At(lin),
                        replicate=DD.At(rep),
                        time=DD.At(last(DD.dims(p_traj, :time))),
                    ]),
                    color=color,
                    markersize=8,
                    marker=:utriangle,
                    strokecolor=:black,
                    strokewidth=1,
                )
                # Translate to front of other trajectories
                translate!(point_init, 0, 0, 100)
                translate!(point_final, 0, 0, 100)
            end # if counter
            # Increment counter
            counter += 1
        end # for rep
    end # for lin

    # Set axis limits
    xlims!(ax_landscape, phenotype_lims.x...)
    ylims!(ax_landscape, phenotype_lims.y...)
end # for i

# ------------------------------------------------------------------------------
# Add axis labels and adjust subplot spacing
# ------------------------------------------------------------------------------

# Loop over rows 
for i in 1:n_rows
    # Adjust column gap
    colgap!.(gl_subrow_dict[i], -5)
    # Add x-axis label for fitness landscape
    Label(
        gl_subrow_dict[i][1][end, :, Bottom()],
        "phenotype 1",
        fontsize=12,
    )
    # Add x-axis label for time series
    Label(
        gl_subrow_dict[i][2][end, :, Bottom()],
        "time",
        fontsize=12,
    )
end

# ------------------------------------------------------------------------------
# Add subplot labels
# ------------------------------------------------------------------------------

# Add subplot labels
Label(
    gl02A_banner[1, 1, Left()], "(A)",
    fontsize=20,
    padding=(0, 15, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl02B_banner[1, 1, Left()], "(B)",
    fontsize=20,
    padding=(0, 0, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Adjust row gap
rowgap!(gl, 5)

# Save figure
save("$(fig_dir)/fig02_v05.pdf", fig)
save("$(fig_dir)/fig02_v05.png", fig)

fig
