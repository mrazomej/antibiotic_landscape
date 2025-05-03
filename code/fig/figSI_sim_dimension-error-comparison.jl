## Description:
# This script plots the comparsion of the reconstruction error between different
# latent space dimensions when trained on simulated data.

## =============================================================================

println("Importing packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import library to handle model
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DataFrames as DF
import JLD2

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

println("Defining directories...")

println("Defining directories...")

# Define version directory
metropolis_dir = "$(git_root())/output/metropolis-hastings_sim"

# Define list of versions
# NOTE:
# - v03: 2D latent space trained on 3D simulated data
# - v08: 3D latent space trained on 3D simulated data
versions = ["v03", "v08"]

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading metadata...")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over versions
for version in versions
    # Define model directory
    vae_dir = "$(metropolis_dir)/$(version)/vae"
    # Define state directory
    state_dir = "$(vae_dir)/model_state"
    # Find model file
    model_file = first(Glob.glob("$(vae_dir)/model*.jld2"[2:end], "/"))
    # List epoch parameters
    model_states = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
    # Load model to extract latent space dimension
    rhvae = JLD2.load(model_file)["model"]
    # Extract latent space dimension
    ldim = size(rhvae.centroids_latent, 1)

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
            :version => version,
            :ldim => ldim,
        )
        # Append temporary dataframe to main dataframe
        global df_meta = DF.vcat(df_meta, df_tmp)
    end # for f in model_states
end # for version in versions

## =============================================================================

println("Plotting error on training and validation sets...")

# ------------------------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------------------------

# Initialize figure
fig = Figure(size=(600, 300))

# Add global grid layout
gl = GridLayout(fig[1, 1])

# Add grid layout for banner
gl_banner = GridLayout(gl[1, 1])
# Add grid layout for axes
gl_axes = GridLayout(gl[2, 1])

# ------------------------------------------------------------------------------
# Add banner
# ------------------------------------------------------------------------------

# Add box for section title
Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-45, right=-5) # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "reconstruction error comparison for 3D simulated adaptive dynamics",
    fontsize=12,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-30) # Moves text to the left
)

# ------------------------------------------------------------------------------
# Add axes
# ------------------------------------------------------------------------------

# Add axis for training error
ax_train = Axis(
    gl_axes[1, 1],
    xlabel="epoch",
    ylabel="mean squared error",
    title="training set",
)

# Add axis for validation error
ax_val = Axis(
    gl_axes[1, 2],
    xlabel="epoch",
    ylabel="mean squared error",
    title="validation set",
)

# Group data by latent space dimension
df_group = DF.groupby(df_meta, :ldim)

# Loop over groups
for df_ldim in df_group
    # Extract latent space dimension
    ldim = df_ldim.ldim[1]
    # Plot training error
    lines!(ax_train, df_ldim.epoch, df_ldim.mse_train, label="$(ldim)D")
    # Plot validation error
    lines!(ax_val, df_ldim.epoch, df_ldim.mse_val, label="$(ldim)D")
end # for (ldim, df_ldim) in df_group

# Add legend
Legend(gl_axes[1, 3], ax_train, "RHVAE\ndim", framevisible=true)

# Save figure
save("$(fig_dir)/figSI_sim_dimension-error-comparison.png", fig)
save("$(fig_dir)/figSI_sim_dimension-error-comparison.pdf", fig)

fig
