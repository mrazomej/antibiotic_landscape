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
fig = Figure(size=(600, 600))

# Add grid layout for Fig01A
gl01A = fig[1:3, 1] = GridLayout()

# Add grid layout for Fig01B
gl01B = fig[4:7, 1] = GridLayout()

# Add grid layout for Fig01C
gl01C = fig[8:10, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Fig01A
# ------------------------------------------------------------------------------

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
M = mh.mutational_landscape(x, y, mutational_landscape)

# Add GridLayout for genetic density
gl_mut = GridLayout(gl01B[1, 1:9])

# Add axis for trajectory in fitness landscape
ax1 = Axis(
    gl_mut[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Genetic Density",
)

# Plot heatmap of mutational landscape
hm_mut = heatmap!(ax1, x, y, M, colormap=:magma)

# Plot contour plot
contour!(ax1, x, y, M, color=:white)

# Add colorbar for mutational landscape
Colorbar(
    gl_mut[1, 2],
    hm_mut,
    label="genetic density (a.u.)",
    height=Relative(3 / 4)
)

# Adjust spacing between plot and colorbar
colgap!(gl_mut, 10)

# ------------------------------------------------------------------------------

# Add GridLayout for fitness
gl_fit = GridLayout(gl01B[1, 11:19])

# Add axis for trajectory in mutational landscape
ax2 = Axis(
    gl_fit[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    aspect=AxisAspect(1),
    title="Fitness",
)

# Plot a heatmap of the fitness landscape
hm_fit = heatmap!(ax2, x, y, F, colormap=:viridis)

# Plot contour plot
contour!(ax2, x, y, F, color=:white)

# Add colorbar for fitness landscape
Colorbar(
    gl_fit[1, 2],
    hm_fit,
    label="fitness (a.u.)",
    height=Relative(3 / 4)
)

# Adjust spacing between plot and colorbar
colgap!(gl_fit, 10)

# ------------------------------------------------------------------------------

# Define points position
p_blue = [0, 0]
p_red = [-2.75, -2.75]

# Plot points on both axes
scatter!.([ax1, ax2], p_blue..., color=Antibiotic.viz.colors()[:blue])
scatter!.([ax1, ax2], p_red..., color=Antibiotic.viz.colors()[:red])


# Add labels to points
text!.(
    [ax1, ax2],
    p_blue + [-0.6, 0.1]...,
    text=L"{p}^{(\text{b})}",
    color=Antibiotic.viz.colors()[:blue]
)

text!.(
    [ax1, ax2],
    p_red .+ [0.1, -0.6]...,
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
G = mh.mutational_landscape(x, y, genetic_density)

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
            yticklabelsvisible=false,
            xticklabelsvisible=false,
        )
    else
        # Axis without y-axis label
        ax = Axis(gl01C[row, col], aspect=AxisAspect(1))
        # Hide axis labels
        hidedecorations!(ax)
    end
    # Hide axis labels
    # Evaluate fitness landscape
    F = mh.fitness(x, y, fit_lan)
    # Plot fitness landscape
    heatmap!(ax, x, y, F, colormap=:viridis)
    # Plot fitness landscape contour plot
    contour!(ax, x, y, F, color=:white)
    # Plot genetic density contour
    contour!(ax, x, y, G, color=:black, linestyle=:dash)

    # Plot points
    scatter!(ax, p_blue..., color=Antibiotic.viz.colors()[:blue])
    scatter!(ax, p_red..., color=Antibiotic.viz.colors()[:red])
    # Add labels to points
    tooltip!(
        ax,
        Point2f(p_blue),
        L"F({p}^{(\text{b})}, E_{%$(i)})",
        textcolor=Antibiotic.viz.colors()[:blue],
        outline_linewidth=1,
        fontsize=12,
        outline_color="#E6E6EF",
    )
    tooltip!(
        ax,
        Point2f(p_red),
        L"F({p}^{(\text{r})}, E_{%$(i)})",
        textcolor=Antibiotic.viz.colors()[:red],
        outline_linewidth=1,
        fontsize=12,
        placement=:right,
        outline_color="#E6E6EF",
    )
end

# Add global x and y labels
Label(gl01C[end+1, :], "phenotype 1")
# Label(gl01C[:, 0], "phenotype 2", rotation=π / 2)

# Adjust column gap
colgap!(gl01C, 3)
# Adjust row gap
rowgap!(gl01C, 0)

# ------------------------------------------------------------------------------
# Subplot labels
# ------------------------------------------------------------------------------

# Add subplot labels
Label(
    gl01A[1, 1, TopLeft()], "(A)",
    fontsize=24,
    padding=(0, 25, 5, 0),
    halign=:right
)

Label(
    gl01B[1, 1, TopLeft()], "(B)",
    fontsize=24,
    padding=(0, 25, -20, 0),
    halign=:right
)

Label(
    gl01C[1, 1, TopLeft()], "(C)",
    fontsize=24,
    padding=(0, 25, -25, 0),
    halign=:right
)

# ------------------------------------------------------------------------------
# Save figure
save("$(fig_dir)/fig01_v01.pdf", fig)

fig
