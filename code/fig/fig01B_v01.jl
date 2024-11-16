## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

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

## =============================================================================

println("Plotting evolution fitness and mutational landscapes...")

# Define range of phenotypes to evaluate
x = range(-4, 4, length=100)
y = range(-4, 4, length=100)

# Create meshgrid
F = mh.fitness(x, y, evolution_condition)
M = mh.genetic_density(x, y, mutational_landscape)

# Initialize figure
fig = Figure(size=(700, 300))

# Add global GridLayout
gl = GridLayout(fig[1, 1])

# Add GridLayout for genetic density
gl_mut = GridLayout(gl[1, 1:9])

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
Colorbar(gl_mut[1, 2], hm_mut, label="genetic density (a.u.)")

# Adjust spacing between plot and colorbar
colgap!(gl_mut, 10)

# ------------------------------------------------------------------------------

# Add GridLayout for fitness
gl_fit = GridLayout(gl[1, 11:19])

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
Colorbar(gl_fit[1, 2], hm_fit, label="fitness (a.u.)")

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
    text=L"{p̲}^{(\text{b})}",
    color=Antibiotic.viz.colors()[:blue]
)

text!.(
    [ax1, ax2],
    p_red .+ [0.1, -0.6]...,
    text=L"{p̲}^{(\text{r})}",
    color=Antibiotic.viz.colors()[:red]
)

# Save figure
save("$(fig_dir)/fig01B.pdf", fig)

fig