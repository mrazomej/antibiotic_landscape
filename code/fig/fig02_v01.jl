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
sim_dir = "$(git_root())/output/metropolis-hastings_sim/v06/sim_evo"

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

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(600, 700))

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

# Add grid layout for fig02C section banner
gl02C_banner = gl[3, :] = GridLayout()
# Add grid layout for Fig02C
gl02C = gl[4, :] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust proportions
# ------------------------------------------------------------------------------

# Adjust row sizes
rowsize!(gl, 2, Auto(1))
rowsize!(gl, 4, Auto(0.85))

# Adjust col sizes
colsize!(gl, 1, Auto(1))
colsize!(gl, 2, Auto(1.5))

# ------------------------------------------------------------------------------
# Section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl02A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50) # Moves box to the left and right
)
Box(
    gl02B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-40, right=-50)
)
Box(
    gl02C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50, right=-50)
)

# Add section title
Label(
    gl02A_banner[1, 1],
    "adaptive dynamics on fitness landscape",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-40) # Moves text to the left
)

Label(
    gl02B_banner[1, 1],
    "alternative landscapes adaptive dynamics",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-30)
)

Label(
    gl02C_banner[1, 1],
    "fitness profiles time-series",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-40)
)

# ------------------------------------------------------------------------------
# Fig02A
# ------------------------------------------------------------------------------

println("Plotting example evolutionary trajectories...")

# Define landscape index
landscape_idx = 1

# Extract data
phenotype_trajectory = fitnotype_profiles.phenotype[
    evo=landscape_idx, landscape=landscape_idx
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
ax = Axis(
    gl02A[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
)

# Plot fitness landscape
heatmap!(ax, x, y, F, colormap=:viridis)

# Plot contour lines for fitness landscape
contour!(ax, x, y, F, color=:black, linestyle=:dash)

# Plot contour lines for genetic density
contour!(ax, x, y, G, color=:white, linestyle=:solid)

# Initialize counter
counter = 1
# Loop over lineages
for lin in DD.dims(phenotype_trajectory, :lineage)
    # Loop over replicates
    for rep in DD.dims(phenotype_trajectory, :replicate)
        # Plot trajectory
        lines!(
            ax,
            phenotype_trajectory[
                phenotype=DD.At(:x1),
                lineage=lin,
                replicate=rep,
            ].data,
            phenotype_trajectory[
                phenotype=DD.At(:x2),
                lineage=lin,
                replicate=rep,
            ].data,
            color=ColorSchemes.glasbey_hv_n256[counter],
            linewidth=1.5
        )
        # Increment counter
        global counter += 1
    end # for rep
end # for lin

# Set axis limits
xlims!(ax, phenotype_lims.x...)
ylims!(ax, phenotype_lims.y...)

# ------------------------------------------------------------------------------
# Fig02B
# ------------------------------------------------------------------------------

# Define number of rows and columns
n_rows = 3
n_cols = 3

# Define limits of phenotype space
phenotype_lims = (
    x=(-5, 5),
    y=(-5, 5),
)

# Define range of phenotypes to evaluate
x = range(phenotype_lims.x..., length=100)
y = range(phenotype_lims.y..., length=100)

# Create meshgrid for genetic density
G = mh.genetic_density(x, y, genetic_density)

# Loop over plots
for i in 1:(n_rows*n_cols)
    # Define row and column
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1

    # Add axis
    ax = Axis(gl02B[row, col], aspect=AxisAspect(1))
    # Remove axis labels
    hidedecorations!(ax)

    # Define landscape index
    landscape_idx = i + 1

    # Extract fitness landscape
    fitness_landscape = fitness_landscapes[landscape_idx]

    # Create meshgrid for fitness landscape
    F = mh.fitness(x, y, fitness_landscape)


    # Plot fitness landscape
    heatmap!(ax, x, y, F, colormap=:viridis)
    # Plot fitness landscape contour lines
    contour!(ax, x, y, F, color=:black, linestyle=:dash, levels=3)

    # Plot genetic density contour lines
    contour!(ax, x, y, G, color=:white, linestyle=:solid, levels=3)

    # Extract data
    phenotype_trajectory = fitnotype_profiles.phenotype[
        evo=landscape_idx, landscape=landscape_idx
    ]

    # Initialize counter
    counter = 1

    # Loop over lineages
    for lin in DD.dims(phenotype_trajectory, :lineage)
        # Loop over replicates
        for rep in DD.dims(phenotype_trajectory, :replicate)
            # Plot trajectory
            lines!(
                ax,
                phenotype_trajectory[
                    phenotype=DD.At(:x1),
                    lineage=lin,
                    replicate=rep,
                ].data,
                phenotype_trajectory[
                    phenotype=DD.At(:x2),
                    lineage=lin,
                    replicate=rep,
                ].data,
                color=ColorSchemes.glasbey_hv_n256[counter],
                linewidth=1
            )
            # Increment counter
            counter += 1
        end # for rep
    end # for lin

    # Set axis limits
    xlims!(ax, phenotype_lims.x...)
    ylims!(ax, phenotype_lims.y...)
end # for i

# Adjust subplot spacing
colgap!(gl02B, 3)
rowgap!(gl02B, 3)

# Add global axis labels
Label(
    gl02B[end, :, Bottom()],
    "phenotype 1",
    fontsize=16,
    padding=(0, 0, -15, 5),
)
Label(
    gl02B[:, 1, Left()],
    "phenotype 2",
    fontsize=16,
    rotation=π / 2,
    padding=(5, 5, 0, 0),
)

# ------------------------------------------------------------------------------
# Fig02C
# ------------------------------------------------------------------------------

println("Plotting fitness profiles over time...")

# Define rows and columns
rows = 2
cols = 4

# Loop through rows and columns
for i in 1:(rows*cols)
    # Define row and column
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1

    # Define axis
    ax = Axis(gl02C[row, col])

    # Add axis title
    if (col == 1) & (row == 1)
        ax.title = "env. $(i) (selection)"
    else
        ax.title = "env. $(i)"
    end # if

    # Set title fontsize
    ax.titlesize = 12

    # Initialize counter
    counter = 1

    # Loop through replicates
    for rep in DD.dims(fitnotype_profiles, :replicate)
        # Loop through lineage
        for lin in DD.dims(fitnotype_profiles, :lineage)
            # Extract fitness profile
            fitness_profile = fitnotype_profiles.fitness[
                evo=1,
                landscape=i,
                replicate=rep,
                lineage=lin
            ]

            # Plot fitness profile
            lines!(
                ax,
                vec(fitness_profile),
                color=ColorSchemes.glasbey_hv_n256[counter],
                linewidth=1
            )

            # Increment counter
            counter += 1
        end # for lineage
    end # for rep
end # for i

# Adjust subplot spacing
# colgap!(gl02C, 3)
rowgap!(gl02C, 3)

# Add global axis labels
Label(
    gl02C[end, :, Bottom()],
    "time (a.u.)",
    fontsize=16,
    padding=(0, 0, -15, 20),
)
Label(
    gl02C[:, 1, Left()],
    "fitness (a.u.)",
    fontsize=16,
    rotation=π / 2,
    padding=(-15, 20, 0, 0),
)

# ------------------------------------------------------------------------------
# Subplot labels
# ------------------------------------------------------------------------------

# Add subplot labels
Label(
    gl02A[1, 1, TopLeft()], "(A)",
    fontsize=20,
    padding=(0, 20, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl02B[1, 1, TopLeft()], "(B)",
    fontsize=20,
    padding=(0, 10, 0, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl02C[1, 1, TopLeft()], "(C)",
    fontsize=20,
    padding=(0, 20, 20, 0),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# Save figure
save("$(fig_dir)/fig02_v01.pdf", fig)

fig

