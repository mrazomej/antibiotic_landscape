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

# Locate current directory
path_dir = pwd()
# Find the path prefix where to put figures
path_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

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

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/output/barbay_kinsler_2020/"

# Load data
df_kinsler = CSV.read(
    "$(data_dir)/kinsler_combined_hyperfitness.csv",
    DF.DataFrame
)

# Define number of environmenst
n_env = length(unique(df_kinsler.env))

# Pivot to extract standardized mean and standard deviation fitness values
df_kinsler_mean = DF.unstack(df_kinsler, :env, :id, :fitness_mean_standard)

# Extract fitness matrix
data_mean = Float32.(Matrix(df_kinsler_mean[:, DF.Not(:env)]))

# Extract dataframe with only metadata
df_meta = unique(
    df_kinsler[:,
        DF.Not(
            :fitness_mean,
            :fitness_std,
            :env,
            :rep,
            :fitness_mean_standard,
            :fitness_std_standard,
            :type
        )
    ])

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

save("$(fig_dir)/rhvae_loss_train.pdf", fig)
save("$(fig_dir)/rhvae_loss_train.png", fig)

fig

## =============================================================================

println("Load model...")

# Load model
rhvae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(rhvae, JLD2.load(df_meta.model_state[end])["model_state"])
# Update metric parameters
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Mapping fitness data to latent space...")

# Initialize dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over ids
for id in unique(df_kinsler.id)
    # Extract data for current id
    data = df_kinsler[df_kinsler.id.==id, :]
    # Run data through encoder
    latent = rhvae.vae.encoder(data.fitness_mean_standard).μ
    # Extract id metadata
    id_meta = df_meta[df_meta.id.==id, :]
    # Add latent coordinates to dataframe
    id_meta.latent1 .= latent[1]
    id_meta.latent2 .= latent[2]
    # Append to main dataframe
    global df_latent = DF.vcat(df_latent, id_meta)
end # for id in unique(df_kinsler.id)

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
    df_latent.latent1,
    df_latent.latent2,
    markersize=8,
)

save("$(fig_dir)/rhvae_latent_space.pdf", fig)
save("$(fig_dir)/rhvae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by gene...")

# Initialize figure
fig = Figure(size=(600, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :gene)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        label=first(data.gene),
        markersize=7,
        color=ColorSchemes.glasbey_hv_n256[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax, labelsize=8, nbanks=3)

save("$(fig_dir)/rhvae_latent_space_gene.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_gene.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by mut_type...")

# Initialize figure
fig = Figure(size=(500, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :mut_type)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        label=first(data.mut_type),
        markersize=7,
        color=ColorSchemes.glasbey_hv_n256[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax, labelsize=8, nbanks=1)

save("$(fig_dir)/rhvae_latent_space_mut_type.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_mut_type.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by class...")

# Initialize figure
fig = Figure(size=(400, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Group dataframe by :env
df_group = DF.groupby(df_latent, :class)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        label=first(data.class),
        markersize=7,
        color=ColorSchemes.glasbey_hv_n256[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax, labelsize=8, nbanks=1)

save("$(fig_dir)/rhvae_latent_space_class.pdf", fig)
save("$(fig_dir)/rhvae_latent_space_class.png", fig)

fig

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent.latent1) - 3.5,
    maximum(df_latent.latent1) + 3.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent.latent2) - 3.5,
    maximum(df_latent.latent2) + 3.5,
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
fig = Figure(size=(400, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
)

# Plot heatmpat of log determinant of metric tensor
hm = heatmap!(
    ax,
    latent1_range,
    latent2_range,
    logdetG,
    colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
)

# Plot latent space
scatter!(
    ax,
    df_latent.latent1,
    df_latent.latent2,
    markersize=5,
    color=(:white, 0.5)
)

# Add colorbar
Colorbar(fig[1, 2], hm, label="√log[det(G̲̲)]")

save("$(fig_dir)/rhvae_latent_space_metric.png", fig)

fig
