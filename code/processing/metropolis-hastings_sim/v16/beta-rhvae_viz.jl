## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

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

# Import basic math
import StatsBase
import Random
Random.seed!(42)

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
using PDFmerger: append_pdf!
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

println("Defining directories...")

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/vae"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading model...")

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a dictionary
rhvae_kwargs = (K=K, ϵ=ϵ, βₒ=βₒ,)

## =============================================================================

# Find model file
model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
# List epoch parameters
model_states = Glob.glob("$(state_dir)/*.jld2"[2:end], "/")

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

save("$(fig_dir)/rhvae_loss.pdf", fig)
save("$(fig_dir)/rhvae_loss.png", fig)

fig

## =============================================================================

println("Loading data into memory...")

# Define the subsampling interval
n_sub = 10

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))
# Define number of training environments
n_env_train = n_env ÷ 2

# Subsample time series and environments
fitnotype_profiles = fitnotype_profiles[
    time=DD.At(t_init:n_sub:t_final),
    landscape=DD.At(1:n_env_train),
    evo=DD.At(1:n_env_train),
]

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

println("Load model...")

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(df_meta.model_state[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Plotting latent space coordinates...")

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Plot latent space
scatter!(
    ax,
    vec(dd_latent[latent=DD.At(:latent1)]),
    vec(dd_latent[latent=DD.At(:latent2)]),
    markersize=5,
)

save("$(fig_dir)/rhvae_latent_space.pdf", fig)
save("$(fig_dir)/rhvae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by lineage...")

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Loop over lineages
for (i, lin) in enumerate(DD.dims(dd_latent, :lineage))
    # Plot latent space
    scatter!(
        ax,
        vec(dd_latent[latent=DD.At(:latent1), lineage=lin]),
        vec(dd_latent[latent=DD.At(:latent2), lineage=lin]),
        markersize=5,
        color=(ColorSchemes.seaborn_colorblind[i], 0.25),
    )
end # for 

save("$(fig_dir)/rhvae_latent_space_lineage.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_lineage.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by time...")

# Initialize figure
fig = Figure(size=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Define color palette
colors = get(
    ColorSchemes.viridis,
    range(0.0, 1.0, length=length(DD.dims(dd_latent, :time)))
)

# Loop over groups
for (i, time) in enumerate(DD.dims(dd_latent, :time))
    # Plot latent space
    scatter!(
        ax,
        vec(dd_latent[latent=DD.At(:latent1), time=DD.At(time)]),
        vec(dd_latent[latent=DD.At(:latent2), time=DD.At(time)]),
        label="$(time)",
        markersize=5,
        color=(colors[i], 0.25),
    )
end # for 

save("$(fig_dir)/rhvae_latent_space_time.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_time.png", fig)

fig

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 250

# Extract latent space ranges
latent1_range = range(
    minimum(dd_latent[latent=DD.At(:latent1)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent1)]) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(dd_latent[latent=DD.At(:latent2)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent2)]) + 1.5,
    length=n_points
)
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

## =============================================================================

println("Plotting latent space metric...")

# Initialize figure
fig = Figure(size=(700, 300))

# Add axis
ax1 = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)
ax2 = Axis(
    fig[1, 2],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Plot heatmpat of log determinant of metric tensor
hm = heatmap!(
    ax1,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

heatmap!(
    ax2,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=ColorSchemes.tokyo,
)

# Plot latent space
scatter!(
    ax2,
    vec(dd_latent[latent=DD.At(:latent1)]),
    vec(dd_latent[latent=DD.At(:latent2)]),
    markersize=6,
    color=(:white, 0.1),
)

# Add colorbar
Colorbar(fig[1, 3], hm, label="√log[det(G̲̲)]")

save("$(fig_dir)/rhvae_latent_space_metric.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_metric.png", fig)

fig