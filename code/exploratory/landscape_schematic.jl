## =============================================================================

# Import project package 
import Antibiotic
# Import Plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set Plotting style
Antibiotic.viz.theme_makie!()

# Import Random library
import Random

## =============================================================================

# Define the domain over which to compute the fitness landscape
x = LinRange(-5, 5, 200)
y = LinRange(-5, 5, 200)

# Create 2D grid arrays for X and Y
X = reshape(x, length(x), 1)  # Column vector of x
Y = reshape(y, 1, length(y))  # Row vector of y

## =============================================================================

"""
    gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)

Create a 2D Gaussian peak or valley.

# Arguments
- `X`: A matrix representing the x-coordinates.
- `Y`: A matrix representing the y-coordinates.
- `x0`: The x-coordinate of the center of the Gaussian.
- `y0`: The y-coordinate of the center of the Gaussian.
- `sigma_x`: The standard deviation of the Gaussian in the x-direction.
- `sigma_y`: The standard deviation of the Gaussian in the y-direction.
- `amplitude`: The amplitude of the Gaussian.

# Returns
- A matrix representing the 2D Gaussian function evaluated at each (X, Y)
  coordinate.
"""
function gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
    # Calculate the exponent term for the Gaussian function
    exp_term = -((X .- x0) .^ 2 ./ (2 * sigma_x^2) .+
                 (Y .- y0) .^ 2 ./ (2 * sigma_y^2))

    # Return the Gaussian function evaluated at each (X, Y) coordinate
    return amplitude .* exp.(exp_term)
end

## =============================================================================

# Initialize the fitness landscape with zeros
Z = zeros(length(x), length(y))

# Parameters for peaks and valleys: (x0, y0, sigma_x, sigma_y, amplitude)
features = [
    # (-2, -2, 1, 1, 1.5),   # Peak at (-2, -2)
    # (2, 2, 1, 1, -1.5),    # Valley at (2, 2)
    (0, 0, 1.5, 1.5, 10),   # Peak at (0, 0)
    (-3, 3, 0.7, 0.7, -5), # Valley at (-3, 3)
    (3, -3, 0.7, 0.7, 5)   # Peak at (3, -3)
]

# Sum the Gaussian functions to build the landscape
for (x0, y0, sigma_x, sigma_y, amplitude) in features
    Z .+= gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
end

## =============================================================================

# Initialize figure
fig = Figure(resolution=(800, 600))

# Add 3D axis 
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="fitness",
    xzpanelcolor=:white,
    xypanelcolor=:white,
    yzpanelcolor=:white,
    elevation=0.085π
)

# Plot a flat surface 
surface!(
    ax, x, y, zeros(length(x), length(y)) .- 15;
    colormap=ColorSchemes.grays,
    shading=false,
)

# Plot the surface of the fitness landscape
surface!(
    ax, x[1:10:end], y[1:10:end], Z'[1:10:end, 1:10:end];
    colormap=:viridis,
    shading=false,
)

wireframe!(
    ax, x[1:10:end], y[1:10:end], Z'[1:10:end, 1:10:end];
    color=(:black, 0.5)
    # colormap=:viridis,
    # shading=false,
)

hidedecorations!(ax)
hidespines!(ax)

# Adjust camera angle for better visualization
# camposition!(ax, Vec3f0(10, -10, 10))
# camlookat!(ax, Vec3f0(0, 0, 0))

# save figure
save("/Users/mrazo/Downloads/landscape_schematic.png", fig)

# Display the plot
fig

## =============================================================================

# Define the domain over which to compute the fitness landscape
x = LinRange(-5, 5, 200)
y = LinRange(-5, 5, 200)

# Create 2D grid arrays for X and Y
X = reshape(x, length(x), 1)  # Column vector of x
Y = reshape(y, 1, length(y))  # Row vector of y

# Parameters for peaks and valleys: (x0, y0, sigma_x, sigma_y, amplitude)
features = [
    # (-2, -2, 1, 1, 1.5),   # Peak at (-2, -2)
    # (2, 2, 1, 1, -1.5),    # Valley at (2, 2)
    (0, 0, 1.5, 1.5, 2),   # Peak at (0, 0)
    (-3, 3, 0.7, 0.7, -1), # Valley at (-3, 3)
    (3, -3, 0.7, 0.7, 1)   # Peak at (3, -3)
]

# Function to create 2D Gaussian peaks and valleys
function gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
    exp_term = -((X .- x0) .^ 2 ./ (2 * sigma_x^2) .+
                 (Y .- y0) .^ 2 ./ (2 * sigma_y^2))
    return amplitude .* exp.(exp_term)
end

# Initialize the fitness landscape with zeros
Z = zeros(length(x), length(y))

# Sum the Gaussian functions to build the landscape
for (x0, y0, sigma_x, sigma_y, amplitude) in features
    Z .+= gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
end

# Transpose Z for correct orientation in the plot
Z = Z'

# Define the fitness function
function fitness(x, y)
    z = 0.0
    for (x0, y0, sigma_x, sigma_y, amplitude) in features
        exp_term = -((x - x0)^2 / (2 * sigma_x^2) +
                     (y - y0)^2 / (2 * sigma_y^2))
        z += amplitude * exp(exp_term)
    end
    return z
end

# Define the gradient of the fitness function
function gradient_fitness(x, y)
    gx = 0.0
    gy = 0.0
    for (x0, y0, sigma_x, sigma_y, amplitude) in features
        exp_term = -((x - x0)^2 / (2 * sigma_x^2) +
                     (y - y0)^2 / (2 * sigma_y^2))
        G = amplitude * exp(exp_term)
        dx = G * (-(x - x0) / (sigma_x^2))
        dy = G * (-(y - y0) / (sigma_y^2))
        gx += dx
        gy += dy
    end
    return gx, gy
end

# Gradient ascent parameters
x_start = -3.0  # Starting x position
y_start = -2.0  # Starting y position
alpha = 0.1     # Learning rate
max_iter = 1000 # Maximum number of iterations
threshold = 1e-6 # Convergence threshold

# Initialize variables for gradient ascent
x_curr = x_start
y_curr = y_start
path = [(x_curr, y_curr)]

# Perform gradient ascent
for i in 1:max_iter
    gx, gy = gradient_fitness(x_curr, y_curr)
    grad_norm = sqrt(gx^2 + gy^2)
    if grad_norm < threshold
        break
    end
    # Update position
    x_curr += alpha * gx
    y_curr += alpha * gy
    # Store path
    push!(path, (x_curr, y_curr))
end

# Extract the path coordinates
x_path = [p[1] for p in path]
y_path = [p[2] for p in path]
z_path = [fitness(x, y) for (x, y) in path]

# Create a figure and axis for the 3D plot
fig = Figure(resolution=(800, 600))
ax = Axis3(fig[1, 1], xlabel="X-axis", ylabel="Y-axis", zlabel="Fitness")

# Plot the surface of the fitness landscape
surface!(ax, x, y, Z, colormap=:viridis, shading=false)

# Plot the gradient ascent path
lines!(ax, x_path, y_path, z_path, color=:red, linewidth=2)

# Adjust camera angle for better visualization
# camposition!(ax, Vec3f0(10, -10, 10))
# camlookat!(ax, Vec3f0(0, 0, 0))

# Display the plot
fig

## =============================================================================

Random.seed!(42)

# Define the domain over which to compute the fitness landscape
x = LinRange(-5, 5, 25)
y = LinRange(-5, 5, 25)

# Create 2D grid arrays for X and Y
X = reshape(x, length(x), 1)  # Column vector of x
Y = reshape(y, 1, length(y))  # Row vector of y

# Parameters for peaks and valleys: (x0, y0, sigma_x, sigma_y, amplitude)
features = [
    (0, 0, 1.5, 1.5, 2),   # Peak at (0, 0)
    (-3, 3, 0.7, 0.7, -1), # Valley at (-3, 3)
    (3, -3, 0.7, 0.7, 1)   # Peak at (3, -3)
]

# Function to create 2D Gaussian peaks and valleys
function gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
    exp_term = -((X .- x0) .^ 2 ./ (2 * sigma_x^2) .+
                 (Y .- y0) .^ 2 ./ (2 * sigma_y^2))
    return amplitude .* exp.(exp_term)
end

# Initialize the fitness landscape with zeros
Z = zeros(length(x), length(y))

# Sum the Gaussian functions to build the landscape
for (x0, y0, sigma_x, sigma_y, amplitude) in features
    Z .+= gaussian2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)
end

# Transpose Z for correct orientation in the plot
Z = Z'

# Define the fitness function
function fitness(x, y)
    z = 0.0
    for (x0, y0, sigma_x, sigma_y, amplitude) in features
        exp_term = -((x - x0)^2 / (2 * sigma_x^2) +
                     (y - y0)^2 / (2 * sigma_y^2))
        z += amplitude * exp(exp_term)
    end
    return z
end

# Define the gradient of the fitness function
function gradient_fitness(x, y)
    gx = 0.0
    gy = 0.0
    for (x0, y0, sigma_x, sigma_y, amplitude) in features
        exp_term = -((x - x0)^2 / (2 * sigma_x^2) +
                     (y - y0)^2 / (2 * sigma_y^2))
        G = amplitude * exp(exp_term)
        dx = G * (-(x - x0) / (sigma_x^2))
        dy = G * (-(y - y0) / (sigma_y^2))
        gx += dx
        gy += dy
    end
    return gx, gy
end

# Gradient ascent parameters
x_start = -3.0  # Starting x position
y_start = -3.0  # Starting y position
alpha = 0.1     # Learning rate
max_iter = 180 # Maximum number of iterations
threshold = 1e-4 # Convergence threshold
noise_level = 0.9 # Tunable noise level (standard deviation of the noise)

# Initialize variables for stochastic gradient ascent
x_curr = x_start
y_curr = y_start
path = [(x_curr, y_curr)]

# Perform stochastic gradient ascent
for i in 1:max_iter
    gx, gy = gradient_fitness(x_curr, y_curr)
    grad_norm = sqrt(gx^2 + gy^2)
    if grad_norm < threshold
        break
    end
    # Add stochastic noise to the gradient
    gx_noisy = gx + noise_level * randn()
    gy_noisy = gy + noise_level * randn()
    # Update position
    x_curr += alpha * gx_noisy
    y_curr += alpha * gy_noisy
    # Store path
    push!(path, (x_curr, y_curr))
end

# Extract the path coordinates
x_path = [p[1] for p in path]
y_path = [p[2] for p in path]
z_path = [fitness(x, y) for (x, y) in path]

# Create a figure and axis for the 3D plot
fig = Figure(resolution=(800, 800))
ax = Axis3(
    fig[1, 1],
    xlabel="phenotype 1",
    ylabel="phenotype 2",
    zlabel="fitness",
    xzpanelcolor=:white,
    xypanelcolor=:white,
    yzpanelcolor=:white,
    elevation=0.085π
)

# Plot the surface of the fitness landscape
surface!(ax, x, y, Z .* 3, colormap=:viridis, shading=false)

# Plot flat surface
wireframe!(
    ax,
    [-5, 5],
    [-5, 5],
    [0 0; 0 0] .- 15;
    color=:black
)

# Plot the stochastic gradient ascent path
lines!(
    ax, x_path, y_path, z_path * 3,
    color=Antibiotic.viz.colors()[:red],
    linewidth=3
)
lines!(
    ax, x_path, y_path, ones(length(z_path)) .- 15,
    color=Antibiotic.viz.colors()[:red],
    linewidth=3
)

hidedecorations!(ax)
hidespines!(ax)

save("/Users/mrazo/Downloads/landscape_schematic_trajectory.pdf", fig)
# save("/Users/mrazo/Downloads/landscape_schematic.pdf", fig)

# Display the plot
fig

