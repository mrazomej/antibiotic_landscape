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

save("$(fig_dir)/vae_loss_train.pdf", fig)
save("$(fig_dir)/vae_loss_train.png", fig)

fig

## =============================================================================

println("Loading data into memory...")

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

## =============================================================================

println("Load model...")

# Load model
vae = JLD2.load(model_file)["model"]
# Load latest model state
Flux.loadmodel!(vae, JLD2.load(df_meta.model_state[end])["model_state"])

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
    latent = vae.encoder(data.logic50_mean_std).µ
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
        )
    )
end # for 
# Map data to latent space

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
    markersize=5,
)

save("$(fig_dir)/vae_latent_space.pdf", fig)
save("$(fig_dir)/vae_latent_space.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by environment...")

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
df_group = DF.groupby(df_latent, :env)

# Loop over groups
for (i, data) in enumerate(df_group)
    # Plot latent space
    scatter!(
        ax,
        data.latent1,
        data.latent2,
        label=first(data.env),
        markersize=7,
        color=ColorSchemes.seaborn_colorblind[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax)

save("$(fig_dir)/vae_latent_space_env.pdf", fig)
save("$(fig_dir)/vae_latent_space_env.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by strain...")

# Initialize figure
fig = Figure(size=(425, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1)
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
        label=first(data.strain),
        markersize=7,
        color=ColorSchemes.Dark2_7[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax)

save("$(fig_dir)/vae_latent_space_strain.pdf", fig)
save("$(fig_dir)/vae_latent_space_strain.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by day...")

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
        label="$(first(data.day))",
        markersize=7,
        color=colors[i],
    )
end # for 

# Add legend
Legend(fig[1, 2], ax, "day", nbanks=3)

save("$(fig_dir)/vae_latent_space_day.pdf", fig)
save("$(fig_dir)/vae_latent_space_day.png", fig)

fig