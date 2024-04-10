##
# Import project environment
import Antibiotic

# Import basic math
import LinearAlgebra
import Random
import Distributions

# Load CairoMakie for plotting
using CairoMakie
# using GLMakie
import ColorSchemes
# Activate backend
CairoMakie.activate!()
# GLMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

# Define home directory
home_dir = git_root()

# Define output directory
output_dir = "./output"

##

# Initialize figure
fig = Figure(size=(350, 350))

# Set axis
ax = Axis3(
    fig[1, 1],
    xlabel="fitness E₁",
    ylabel="fitness E₂",
    zlabel="fitness E₃",
    aspect=(1, 1, 1),
    azimuth=1,
)

# Define coordinates of point
point = repeat([0.75], 3)

# Add scatter plot
scatter!(
    ax,
    [point[1]],
    [point[2]],
    [point[3]],
    color=ColorSchemes.seaborn_colorblind[1]
)
# Add line plot
lines!(
    ax,
    [0, point[1]],
    [0, point[2]],
    [0, point[3]],
    color=ColorSchemes.seaborn_colorblind[1]
)
# Add text
text!(
    ax,
    [point[1]],
    [point[2]],
    [point[3]],
    text="mutant Mⱼ",
)

# Set axes limits
xlims!(ax, 0.0, 1.0)
ylims!(ax, 0.0, 1.0)
zlims!(ax, 0.0, 1.0)

# Set camera angles
ax.azimuth = 1
ax.elevation = π / 8

save("$output_dir/3D_fitness_space.png", fig)
save("$output_dir/3D_fitness_space.pdf", fig)

fig

##

#%% Plot 2D plane in 3D space with sparse mutants

# Initialize figure
fig = Figure(size=(350, 350))

# Set axis
ax = Axis3(
    fig[1, 1],
    xlabel="fitness E₁",
    ylabel="fitness E₂",
    zlabel="fitness E₃",
    aspect=(1, 1, 1),
    azimuth=1,
)

Random.seed!(42)
# Define matrix E
# E_mat = rand(3, 2)
E_mat = [
    1.0 0.0
    -0.5 1.0
    -1.0 1.0
]

# Define "expansion factor" for random 
expansion_fac = 0.6
# Define 2D Plane coordinates
coord_2d_plane = [-1 -1; 1 -1; 1 1; -1 1] .* expansion_fac

# Define vertices of 3D plane
vertices = coord_2d_plane * E_mat'

# Define which P_mat in the triangle are connected. Every polygon is made of
# connected triangles
faces = [1 2 3; 3 4 1]

# Plot simplex as mesh with transparency
mesh!(
    ax,
    vertices,
    faces,
    color=("#0173b2", 0.2),
    transparency=true
)

# Add grid lines
lines!(ax, [-1, 1], [0, 0], [0, 0], color=:black)
lines!(ax, [0, 0], [-1, 1], [0, 0], color=:black)
lines!(ax, [0, 0], [0, 0], [-1, 1], color=:black)

# Define number of random mutans
n_mut = 8

# Generate random performances
mut_per = rand(Distributions.Uniform(-1, 1), n_mut, 2) .* expansion_fac

# Convert performance to fitness
mut_fit = mut_per * E_mat'

# Scatter mutants
scatter!(
    ax,
    mut_fit[:, 1],
    mut_fit[:, 2],
    mut_fit[:, 3],
    color=1:n_mut,
    colormap=:thermal
)

xlims!(ax, -1, 1)
ylims!(ax, -1, 1)
zlims!(ax, -1, 1)

ax.azimuth = -0.4
ax.elevation = π / 8
save("$output_dir/3D_fitness_plane.png", fig)
save("$output_dir/3D_fitness_plane.pdf", fig)

fig

##

fig = Figure(size=(350, 350))

# Set axis
ax = Axis3(
    fig[1, 1],
    xlabel="fitness E₁",
    ylabel="fitness E₂",
    zlabel="fitness E₃",
    aspect=(1, 1, 1),
    azimuth=1,
)

# Define 2D Plane coordinates
coord_2d_plane = [-1 -1; 1 -1; 1 1; -1 1] .* expansion_fac

# Define vertices of 3D plane
vertices = coord_2d_plane * E_mat'

# Define which P_mat in the triangle are connected. Every polygon is made of
# connected triangles
faces = [1 2 3; 3 4 1]

# Plot simplex as mesh with transparency
mesh!(
    ax,
    vertices,
    faces,
    color=("#0173b2", 0.2),
    transparency=true
)

# Add grid lines for original axis
lines!(ax, [-1, 1], [0, 0], [0, 0], color=:gray, linestyle=:dash)
lines!(ax, [0, 0], [-1, 1], [0, 0], color=:gray, linestyle=:dash)
lines!(ax, [0, 0], [0, 0], [-1, 1], color=:gray, linestyle=:dash)

# Compute SVD of matrix to get orthogonal vectors
U_mat, S_mat, V_mat = LinearAlgebra.svd(E_mat)

# Add grid lines for new axis
lines!(
    ax,
    [U_mat[1, 1] * 1.5, U_mat[1, 1] * -1.5],
    [U_mat[2, 1] * 1.5, U_mat[2, 1] * -1.5],
    [U_mat[3, 1] * 1.5, U_mat[3, 1] * -1.5],
    color=:red
)

lines!(
    ax,
    [U_mat[1, 2] * 1.5, U_mat[1, 2] * -1.5],
    [U_mat[2, 2] * 1.5, U_mat[2, 2] * -1.5],
    [U_mat[3, 2] * 1.5, U_mat[3, 2] * -1.5],
    color=:red
)

# Compute cross product for third axis
z_new = LinearAlgebra.cross(U_mat[:, 1], U_mat[:, 2])
lines!(
    ax,
    [z_new[1] * 1.5, z_new[1] * -1.5],
    [z_new[2] * 1.5, z_new[2] * -1.5],
    [z_new[3] * 1.5, z_new[3] * -1.5],
    color=:red
)

xlims!(ax, -1, 1)
ylims!(ax, -1, 1)
zlims!(ax, -1, 1)

ax.azimuth = -0.4
ax.elevation = π / 8

save("$output_dir/3D_fitness_plane_new_coord.png", fig)
save("$output_dir/3D_fitness_plane_new_coord.pdf", fig)

fig

##

#%% Generating matrix F for several mutants and environments %##

Random.seed!(42)

# Define numebr of dimensions on simplex
n_traits = 2

# Define number of mutants
n_mut = 500

# Define number of environments
n_env = 50

# Sample mutant phenotypes uniformly on simplex
P_mat = rand(Distributions.Dirichlet(repeat([1], n_traits)), n_mut)'

# Sample environment weights for each phenotype
E_mat = rand(Distributions.Dirichlet(repeat([1], n_traits)), n_env)

# Compute fitness matrix
F_mat = P_mat * E_mat

# Compute SVD on F matrix
U_mat, Σ_mat, V_mat = LinearAlgebra.svd(F_mat)

# Initialize figure
fig = Figure(size=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="singular value index",
    ylabel="singular value",
    title="linear model",
    yscale=log10
)

# Add inset
ax_inset = Axis(
    fig[1, 1],
    yscale=log10,
    width=Relative(0.4),
    height=Relative(0.4),
    halign=0.9,
    valign=0.9,
    backgroundcolor=("#FFEDCE", 1)
)

# Plot line
lines!(ax, Σ_mat)
# Plot points
scatter!(ax, Σ_mat, markersize=6)

# Plot line
lines!(ax_inset, Σ_mat[1:n_traits+1])
# Plot points
scatter!(ax_inset, Σ_mat[1:n_traits+1], markersize=6)
# Set axis limit
ylims!(ax_inset, low=1E-3)

save("$output_dir/SVD_linear_model.png", fig)
save("$output_dir/SVD_linear_model.pdf", fig)