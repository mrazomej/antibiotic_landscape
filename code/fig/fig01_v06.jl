## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

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

# Define output directory
fig_dir = "$(git_root())/fig/main"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Setting global layout...")

# Initialize figure
fig = Figure(size=(600, 650))

# ------------------------------------------------------------------------------
# Plot layout
# ------------------------------------------------------------------------------

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for fig01A section banner
gl01A_banner = gl[1, 1:5] = GridLayout()
# Add grid layout for Fig01A
gl01A = gl[2, 1:5] = GridLayout()

# Add grid layout for fig01B section banner
gl01B_banner = gl[3, 1:5] = GridLayout()
# Add grid layout for Fig01B
gl01B = gl[4, 1:5] = GridLayout()

# Add grid layout for fig01C section banner
gl01C_banner = gl[5, 1:5] = GridLayout()
# Add grid layout for Fig01C
gl01C = gl[6, 1:4] = GridLayout()

# Add grid layout for fig01D section banner
gl01D = gl[6, 5] = GridLayout()

# ------------------------------------------------------------------------------
# Adjust proportions
# ------------------------------------------------------------------------------

# Adjust row sizes for banners
rowsize!(gl, 1, Auto(1))
rowsize!(gl, 3, Auto(1))
rowsize!(gl, 5, Auto(1))

# Adjust row sizes for plots
rowsize!(gl, 2, Auto(10))
rowsize!(gl, 4, Auto(10))
rowsize!(gl, 6, Auto(10))

# ------------------------------------------------------------------------------
# Section banners
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl01A_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50, right=-50) # Moves box to the left and right
)
Box(
    gl01B_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50, right=-50)
)
Box(
    gl01C_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-50, right=-50)
)

# Add section title
Label(
    gl01A_banner[1, 1],
    "schematic genotype-phenotype-fitness map",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
)

Label(
    gl01B_banner[1, 1],
    "genotype-to-phenotype density and fitness functions on phenotypic space",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-30)
)

Label(
    gl01C_banner[1, 1],
    "fitness functions in multiple environments as experimental observables",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false,
    alignmode=Mixed(; left=-30)
)

# ------------------------------------------------------------------------------
# Fig01A
# ------------------------------------------------------------------------------

# To be added in illustrator

# ------------------------------------------------------------------------------
# Fig01B
# ------------------------------------------------------------------------------

println("Defining phenotypic space...")

# Evolution condition amplitude
fit_evo_amplitude = 5.0
# Evolution condition mean
fit_evo_mean = [0.0, 0.0]
# Evolution condition covariance
fit_evo_covariance = 3.0
# Create peak
evolution_condition = mh.GaussianPeak(
    fit_evo_amplitude,
    fit_evo_mean,
    fit_evo_covariance
)

# Mutational peak amplitude
mut_evo_amplitude = 1.0
# Mutational peak means
mut_means = [
    [-1.5, -1.5],
    [1.5, -1.5],
    [1.5, 1.5],
    [-1.5, 1.5],
]
# Mutational peak covariance
mut_evo_covariance = 0.45
# Create mutational peaks
mutational_landscape = mh.GaussianPeaks(
    mut_evo_amplitude,
    mut_means,
    mut_evo_covariance
)

# ------------------------------------------------------------------------------

println("Plotting evolution fitness and mutational landscapes...")

# Define range of phenotypes to evaluate
x = range(-4, 4, length=100)
y = range(-4, 4, length=100)

# Create meshgrid
F = mh.fitness(x, y, evolution_condition)
M = mh.genetic_density(x, y, mutational_landscape)

# Add GridLayout for genetic density
gl_mut = GridLayout(gl01B[1, 1])

# Add axis for trajectory in fitness landscape
ax1 = Axis(
    gl_mut[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Genotype-to-Phenotype\nDensity",
    titlesize=13,
    xlabelsize=13,
    ylabelsize=13,
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot heatmap of mutational landscape
hm_mut = heatmap!(ax1, x, y, M, colormap=Reverse(ColorSchemes.Purples_9))

# Plot contour plot
contour!(ax1, x, y, M, color=:white)

# Add colorbar for mutational landscape
Colorbar(
    gl_mut[1, 2],
    hm_mut,
    label="genotype-phenotype\ndensity",
    labelsize=13,
    height=Relative(1),
    ticksvisible=false,
    ticklabelsvisible=false
)

# Adjust spacing between plot and colorbar
colgap!(gl_mut, -50)

# ------------------------------------------------------------------------------

# Add GridLayout for fitness
gl_fit = GridLayout(gl01B[1, 2])

# Add axis for trajectory in mutational landscape
ax2 = Axis(
    gl_fit[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Fitness",
    titlesize=13,
    xlabelsize=13,
    ylabelsize=13,
    yticklabelsvisible=false,
    xticklabelsvisible=false,
)

# Plot a heatmap of the fitness landscape
hm_fit = heatmap!(ax2, x, y, F, colormap=:algae)

# Plot contour plot
contour!(ax2, x, y, F, color=:white)

# Add colorbar for fitness landscape
Colorbar(
    gl_fit[1, 2],
    hm_fit,
    label="fitness",
    labelsize=13,
    height=Relative(1),
    ticksvisible=false,
    ticklabelsvisible=false,
)

# Adjust spacing between plot and colorbar
colgap!(gl_fit, -50)

# ------------------------------------------------------------------------------

# Define points position
p_blue = [0, 0]
p_red = [-2.75, -2.75]

# Plot points on both axes
scatter!.(
    [ax1, ax2],
    p_blue...,
    color=Antibiotic.viz.colors()[:light_blue],
    strokecolor=Antibiotic.viz.colors()[:dark_blue],
    strokewidth=1.5,
    label=L"{p}^{(\text{b})}"
)
scatter!.(
    [ax1, ax2],
    p_red...,
    color=Antibiotic.viz.colors()[:light_red],
    strokecolor=Antibiotic.viz.colors()[:dark_red],
    strokewidth=1.5,
    label=L"{p}^{(\text{r})}"
)

# Add labels to points
text!(
    ax1,
    p_blue + [-0.8, -0.1]...,
    text=L"{p}^{(\text{b})}",
    color=Antibiotic.viz.colors()[:dark_blue],
)
text!(
    ax2,
    p_blue + [-0.8, -0.1]...,
    text=L"{p}^{(\text{b})}",
    color=:white
)

text!.(
    [ax1, ax2],
    p_red .+ [0.1, -1]...,
    text=L"{p}^{(\text{r})}",
    color=Antibiotic.viz.colors()[:red]
)


# ------------------------------------------------------------------------------
# Fig01C
# ------------------------------------------------------------------------------

println("Defining landscape parameters...")

# Phenotype space dimensionality
n_dim = 2

# Define number of fitness landscapes
n_fit_lans = 3

# Define range of peak means
peak_mean_min = -4.0
peak_mean_max = 4.0

# Define range of fitness amplitudes
fit_amp_min = 1.0
fit_amp_max = 5.0

# Define covariance range
fit_cov_min = 0.5
fit_cov_max = 3.0

# Define possible number of fitness peaks
n_fit_peaks_min = 2
n_fit_peaks_max = 4

# ------------------------------------------------------------------------------

println("Defining phenotypic space...")

# Evolution condition amplitude
fit_evo_amplitude = 5.0
# Evolution condition mean
fit_evo_mean = [0.0, 0.0]
# Evolution condition covariance
fit_evo_covariance = 3.0
# Create peak
evolution_condition = mh.GaussianPeak(
    fit_evo_amplitude,
    fit_evo_mean,
    fit_evo_covariance
)

# Mutational peak amplitude
mut_evo_amplitude = 1.0
# Mutational peak means
mut_means = [
    [-1.5, -1.5],
    [1.5, -1.5],
    [1.5, 1.5],
    [-1.5, 1.5],
]
# Mutational peak covariance
mut_evo_covariance = 0.45
# Create mutational peaks
genetic_density = mh.GaussianPeaks(
    mut_evo_amplitude,
    mut_means,
    mut_evo_covariance
)

# ------------------------------------------------------------------------------

Random.seed!(1)

println("Generating alternative fitness landscapes...")

# Initialize array to hold fitness landscapes
fit_lans = Array{mh.AbstractPeak}(undef, n_fit_lans + 1)

# Store evolution condition in first landscape
fit_lans[1] = evolution_condition

# Loop over number of fitness landscapes
for i in 1:n_fit_lans
    # Sample number of fitness peaks
    n_fit_peaks = rand(n_fit_peaks_min:n_fit_peaks_max)

    # Sample fitness means as 2D vectors from uniform distribution
    fit_means = [
        rand(Distributions.Uniform(peak_mean_min, peak_mean_max), 2)
        for _ in 1:n_fit_peaks
    ]

    # Sample fitness amplitudes from uniform distribution
    fit_amplitudes = rand(
        Distributions.Uniform(fit_amp_min, fit_amp_max), n_fit_peaks
    )

    # Sample fitness covariances from uniform distribution
    fit_covariances = rand(
        Distributions.Uniform(fit_cov_min, fit_cov_max), n_fit_peaks
    )

    # Check dimensionality
    if n_fit_peaks == 1
        # Create fitness peaks
        fit_lans[i+1] = mh.GaussianPeak(
            first(fit_amplitudes), first(fit_means), first(fit_covariances)
        )
    else
        # Create fitness peaks
        fit_lans[i+1] = mh.GaussianPeaks(
            fit_amplitudes, fit_means, fit_covariances
        )
    end # if
end # for


# ------------------------------------------------------------------------------

# Define points position
p_blue = [0, 0]
p_red = [-2.75, -2.75]

# ------------------------------------------------------------------------------

println("Plotting example alternative fitness landscapes...")

# Define number of rows and columns
n_rows = 1
n_cols = length(fit_lans)

# Define ranges of phenotypes to evaluate
x = range(-4, 4, length=100)
y = range(-4, 4, length=100)

# Compute genetic density
G = mh.genetic_density(x, y, genetic_density)

# Loop over fitness landscapes
for i in 1:(n_rows*n_cols)
    # Extract fitness landscape
    fit_lan = fit_lans[i]
    # Define row and column
    row = (i - 1) ÷ n_cols + 1
    col = (i - 1) % n_cols + 1
    # Add axis
    if i == 1
        # Axis with y-axis label
        ax = Axis(
            gl01C[row, col],
            aspect=AxisAspect(1),
            ylabel="phenotype 2",
            ylabelsize=13,
            yticklabelsvisible=false,
            xticklabelsvisible=false,
            title="env. $i (selection)",
            titlesize=13
        )
    else
        ax = Axis(
            gl01C[row, col],
            aspect=AxisAspect(1),
            title="env. $i",
            titlesize=13
        )
        # Hide axis labels
        hidedecorations!(ax)
    end

    # Evaluate fitness landscape
    local F = mh.fitness(x, y, fit_lan)
    # Plot fitness landscape
    if i == 1
        # For evolution condition
        heatmap!(ax, x, y, F, colormap=:algae)
    else
        # For alternative fitness landscapes
        heatmap!(ax, x, y, F, colormap=Reverse(:grays))
    end # if
    # Plot fitness landscape contour
    contour!(
        ax, x, y, F, color=Antibiotic.viz.colors()[:black], linestyle=:dash
    )
    # Plot genetic density contour
    contour!(
        ax,
        x, y, G,
        color=:white,
        linestyle=:solid
    )

    # Plot points
    scatter!(
        ax,
        p_blue...,
        color=Antibiotic.viz.colors()[:light_blue],
        strokecolor=Antibiotic.viz.colors()[:dark_blue],
        strokewidth=1.5,
        label=L"{p}^{(\text{b})}"
    )
    scatter!(
        ax,
        p_red...,
        color=Antibiotic.viz.colors()[:light_red],
        strokecolor=Antibiotic.viz.colors()[:dark_red],
        strokewidth=1.5,
        label=L"{p}^{(\text{r})}"
    )
end

# Add global x and y labels
Label(gl01C[end+1, :], "phenotype 1", fontsize=13)

# Adjust column gap
colgap!(gl01C, 3)
# Adjust row gap
rowgap!(gl01C, 3)

# ------------------------------------------------------------------------------
# Fig01D
# ------------------------------------------------------------------------------

println("Plotting fitness profiles...")

# Evaluate fitness profiles
f_red = mh.fitness.(Ref(p_red), fit_lans)
f_blue = mh.fitness.(Ref(p_blue), fit_lans)

# Initialize axis
ax = Axis(
    gl01D[1, 1],
    ylabel="fitness",
    aspect=AxisAspect(1 / 1.25),
    xgridvisible=false,
    ygridvisible=false,
    yticklabelsvisible=false,
    xticks=(1:length(fit_lans), ["env. $i" for i in 1:length(fit_lans)]),
    xticklabelrotation=π / 2,
    xticklabelsize=12,
    title="fitness profiles\n(observables)",
    titlesize=13,
    ylabelsize=13
)

# Plot fitness profiles
scatterlines!(
    ax,
    f_red,
    color=Antibiotic.viz.colors()[:light_red],
    strokecolor=Antibiotic.viz.colors()[:dark_red],
    strokewidth=1.5,
    linewidth=2
)
scatterlines!(
    ax,
    f_blue,
    color=Antibiotic.viz.colors()[:light_blue],
    strokecolor=Antibiotic.viz.colors()[:dark_blue],
    strokewidth=1.5,
    linewidth=2
)

# ------------------------------------------------------------------------------
# Subplot labels
# ------------------------------------------------------------------------------

# Add subplot labels
Label(
    gl01A[0, 0, TopLeft()], "(A)",
    fontsize=20,
    padding=(0, 10, 0, -10),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl01B[1, 1, TopLeft()], "(B)",
    fontsize=20,
    padding=(0, 10, 0, -45),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl01C[1, 1, TopLeft()], "(C)",
    fontsize=20,
    padding=(0, 10, 0, -60),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

Label(
    gl01D[1, 1, TopLeft()], "(D)",
    fontsize=20,
    padding=(0, 14, 0, -45),
    halign=:right,
    tellwidth=false,
    tellheight=false
)

# ------------------------------------------------------------------------------
# Save figure
# ------------------------------------------------------------------------------

save("$(fig_dir)/fig01_v06.pdf", fig)

fig
