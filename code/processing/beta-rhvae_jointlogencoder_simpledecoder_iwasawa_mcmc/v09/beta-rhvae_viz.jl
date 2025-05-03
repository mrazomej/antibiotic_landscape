## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import CSV
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Import WGLMakie for interactive plotting
using WGLMakie
import ColorSchemes
import Colors
using Bonito
# Activate backend
WGLMakie.activate!()
# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()
# Find the path prefix where to put figures
path_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"
# Define output directory
out_dir = "$(git_root())/output$(path_prefix)"
# Define model directory
model_dir = "$(git_root())/output$(path_prefix)/model_state"

# Define figure directory
fig_dir = "$(git_root())/fig$(path_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = (K=K, ϵ=ϵ, βₒ=βₒ,)

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(out_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = Glob.glob("$(model_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over files
for f in model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", f).captures[1])
    # Load model_state file
    f_load = JLD2.load(f)
    # Extract values
    loss_train = f_load["loss_train"]
    loss_val = f_load["loss_val"]
    mse_train = f_load["mse_train"]
    mse_val = f_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => model_file,
        :model_state => f,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for f in model_states

## =============================================================================

println("Plotting training loss...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="epoch",
    ylabel="loss",
)

# Plot training loss
lines!(
    ax,
    df_meta.epoch,
    df_meta.loss_train,
    label="train",
)
# Plot validation loss
lines!(
    ax,
    df_meta.epoch,
    df_meta.loss_val,
    label="validation",
)

# Add legend
axislegend(ax, position=:rt)

# Add axis
ax = Axis(
    fig[1, 2],
    xlabel="epoch",
    ylabel="MSE",
)

# Plot training loss
lines!(
    ax,
    df_meta.epoch,
    df_meta.mse_train,
    label="train",
)
# Plot validation loss
lines!(
    ax,
    df_meta.epoch,
    df_meta.mse_val,
    label="validation",
)

# Add legend
axislegend(ax, position=:rt)

# Save figure
save("$(fig_dir)/rhvae_training_loss.png", fig)

## =============================================================================

println("Loading data into memory...")

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

## =============================================================================

println("Load model...")

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(df_meta.model_state[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for data in df_group
    # Sort data by drug
    DF.sort!(data, :drug)
    # Run :logic50_mean_std through encoder
    latent = rhvae.vae.encoder(data.logic50_mean_std).µ
    # Append latent coordinates to dataframe
    DF.append!(
        df_latent,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent[1, :],
            :latent2 => latent[2, :],
            :latent3 => latent[3, :],
        )
    )
end # for 

## =============================================================================

println("Plotting latent space coordinates...")

# Initialize figure
fig = Figure(size=(400, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Plot latent space
scatter!(
    ax,
    df_latent.latent1,
    df_latent.latent2,
    df_latent.latent3,
    markersize=5,
)

# Create HTML file with interactive plot
open("$(fig_dir)/latent_space_3d.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Plotting latent space coordinates colored by environment...")

# Initialize figure
fig = Figure(size=(500, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :env)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        data.latent3,
        label=first(data.env),
        markersize=5,
        color=ColorSchemes.seaborn_colorblind[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax)

# Create HTML file with interactive plot
open("$(fig_dir)/latent_space_3d_colored_by_env.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Plotting latent space coordinates colored by strain...")

# Initialize figure
fig = Figure(size=(500, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :strain)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        data.latent3,
        label=first(data.strain),
        markersize=5,
        color=ColorSchemes.Dark2_7[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax)

# Create HTML file with interactive plot
open("$(fig_dir)/latent_space_3d_colored_by_strain.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Plotting latent space coordinates colored by day...")

# Initialize figure
fig = Figure(size=(600, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :day)

# Define color palette
colors = get(ColorSchemes.Blues_9, range(0.5, 1.0, length=length(df_group)))

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        data.latent3,
        label="$(first(data.day))",
        markersize=5,
        color=colors[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax, "day", nbanks=3)

# Create HTML file with interactive plot
open("$(fig_dir)/latent_space_3d_colored_by_day.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent.latent1) - 1,
    maximum(df_latent.latent1) + 1,
    length=n_points
)
latent2_range = range(
    minimum(df_latent.latent2) - 1,
    maximum(df_latent.latent2) + 1,
    length=n_points
)
latent3_range = range(
    minimum(df_latent.latent3) - 1,
    maximum(df_latent.latent3) + 1,
    length=n_points
)

# Define latent ranges
latent_ranges = (latent1_range, latent2_range, latent3_range)

# Define grid points
grid_points = Iterators.product(latent_ranges...)

# Compute inverse metric tensor
Ginv = map(
    point -> AET.RHVAEs.G_inv([point...], rhvae),
    grid_points
)

# Compute metric 
logdetG = map(
    Ginv -> -1 / 2 * AET.utils.slogdet(Ginv),
    Ginv
)

## =============================================================================

println("Plotting latent space metric as contour...")

# Initialize figure
fig = Figure(size=(500, 500))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Plot contour
contour!(
    ax,
    collect(latent_ranges[1]),
    collect(latent_ranges[2]),
    collect(latent_ranges[3]),
    logdetG,
    alpha=0.05,
    levels=7,
    colormap=ColorSchemes.tokyo,
)

# Save figure
# Create HTML file with interactive plot
open("$(fig_dir)/rhvae_latent_space_metric_contour.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Plotting latent space metric as contour with data points...")

# Initialize figure
fig = Figure(size=(500, 500))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Plot contour
contour!(
    ax,
    collect(latent_ranges[1]),
    collect(latent_ranges[2]),
    collect(latent_ranges[3]),
    logdetG,
    alpha=0.05,
    levels=7,
    colormap=ColorSchemes.tokyo,
)

# Plot data points
scatter!(
    ax,
    df_latent.latent1,
    df_latent.latent2,
    df_latent.latent3,
    markersize=5,
    color=(:white, 1),
)

# Create HTML file with interactive plot
open("$(fig_dir)/rhvae_latent_space_metric_contour_with_data.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end

## =============================================================================

println("Plotting latent space metric as contour colored by strain...")

# Initialize figure
fig = Figure(size=(500, 400))

# Add axis
ax = Axis3(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    zlabel="latent dimension 3",
    aspect=(1, 1, 1),
)

# Plot contour
contour!(
    ax,
    collect(latent_ranges[1]),
    collect(latent_ranges[2]),
    collect(latent_ranges[3]),
    logdetG,
    alpha=0.05,
    levels=7,
    colormap=ColorSchemes.tokyo,
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :strain)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        data.latent3,
        label=first(data.strain),
        markersize=5,
        color=ColorSchemes.Dark2_7[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax)

# Create HTML file with interactive plot
open("$(fig_dir)/rhvae_latent_space_metric_contour_colored_by_strain.html", "w") do io
    println(io, "<html><head></head><body>")
    app = App() do
        Card(fig; height="fit-content", width="fit-content")
    end
    show(io, MIME"text/html"(), app)
    println(io, "</body></html>")
end