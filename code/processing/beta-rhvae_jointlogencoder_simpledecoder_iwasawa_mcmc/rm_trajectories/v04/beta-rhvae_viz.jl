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
    # Extract evolution condition removed
    evo = match(r"rhvae_([^_]+)_epoch", f).captures[1]
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
        :evo => evo,
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

# Group dataframe by :evo
df_group = DF.groupby(df_meta, :evo)

# Initialize figure
fig = Figure(size=(600, 300 * length(df_group)))

# Loop through each group
for (i, df_evo) in enumerate(df_group)
    # Add axis for loss
    ax_loss = Axis(
        fig[i, 1],
        xlabel="epoch",
        ylabel="loss",
        title="Evolution condition excluded: $(df_evo.evo[1])"
    )

    # Plot training loss
    lines!(
        ax_loss,
        df_evo.epoch,
        df_evo.loss_train,
        label="train",
    )
    # Plot validation loss
    lines!(
        ax_loss,
        df_evo.epoch,
        df_evo.loss_val,
        label="validation",
    )

    # Add legend
    axislegend(ax_loss, position=:rt)

    # Add axis for MSE
    ax_mse = Axis(
        fig[i, 2],
        xlabel="epoch",
        ylabel="MSE",
    )

    # Plot training MSE
    lines!(
        ax_mse,
        df_evo.epoch,
        df_evo.mse_train,
        label="train",
    )
    # Plot validation MSE
    lines!(
        ax_mse,
        df_evo.epoch,
        df_evo.mse_val,
        label="validation",
    )

    # Add legend
    axislegend(ax_mse, position=:rt)
end

# Save figure
save("$(fig_dir)/rhvae_loss_train.pdf", fig)
save("$(fig_dir)/rhvae_loss_train.png", fig)

fig

## =============================================================================

println("Loading data into memory...")

# Load logic50 data 
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Extract strain and evolution condition from :env by splitting by _
DF.insertcols!(
    df_logic50,
    :strain => getindex.(split.(df_logic50.env, "_"), 1),
    :evo => getindex.(split.(df_logic50.env, "_"), 3),
)

## =============================================================================

println("Load models...")

# Initialize dictionary to store models
rhvae_models = Dict()

# Loop through each unique evolution condition
for evo in unique(df_meta.evo)
    # Get the latest model state file for this evolution condition
    latest_model_state = df_meta[df_meta.evo.==evo, :model_state][end]
    # Load base model
    rhvae = JLD2.load(model_file)["model"]
    # Load latest model state for this evolution condition
    Flux.loadmodel!(rhvae, JLD2.load(latest_model_state)["model_state"])
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae)
    # Store the model in the dictionary
    rhvae_models[evo] = rhvae
end

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for (i, data) in enumerate(df_group)
    # Sort data by drug
    DF.sort!(data, :drug)
    # Loop over each model (one for each evolution condition)
    for (evo, rhvae) in rhvae_models
        # Run :logic50_mean_std through encoder
        latent = rhvae.vae.encoder(data.logic50_mean_std).µ
        # Determine if data is in training or validation set for this model
        train = evo ≠ first(data.evo)
        # Append latent coordinates to dataframe
        DF.append!(
            df_latent,
            DF.DataFrame(
                :day .=> first(data.day),
                :strain_num .=> first(data.strain_num),
                :meta .=> first(data.env),
                :evo .=> first(data.evo),
                :strain .=> first(data.strain),
                :latent1 => latent[1, :],
                :latent2 => latent[2, :],
                :train => train,
                :model_evo => evo
            )
        )
    end # for rhvae_models
end # for df_group

## =============================================================================

println("Plotting latent space coordinates for each model...")

# Initialize figure
fig = Figure(size=(650, 650))

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(fig[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]
    scatter!(
        axes[row, col],
        data.latent1,
        data.latent2,
        markersize=5,
    )
end

save("$(fig_dir)/rhvae_latent_spaces.pdf", fig)
save("$(fig_dir)/rhvae_latent_spaces.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by environment for each model...")

# Initialize figure
fig = Figure(size=(800, 650))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(gl[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Group dataframe by :env for this model
    df_group = DF.groupby(data, :evo)

    # Loop over environment groups
    for (i, env_data) in enumerate(df_group)
        # Plot latent space
        scatter!(
            axes[row, col],
            env_data.latent1,
            env_data.latent2,
            label=first(env_data.evo),
            markersize=5,
            color=ColorSchemes.seaborn_colorblind[i],
        )
    end # for env_data

    # Check if is the first plot
    if idx == 1
        # Add legend to each subplot
        Legend(gl[:, col+2], axes[row, col], "Evolution\ncondition")
    end # if
end # for

save("$(fig_dir)/rhvae_latent_spaces_env.pdf", fig)
save("$(fig_dir)/rhvae_latent_spaces_env.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by strain...")

# Initialize figure
fig = Figure(size=(800, 650))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(gl[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Group dataframe by :strain for this model
    df_group = DF.groupby(data, :strain)

    # Loop over strain groups
    for (i, strain_data) in enumerate(df_group)
        # Plot latent space
        scatter!(
            axes[row, col],
            strain_data.latent1,
            strain_data.latent2,
            label=first(strain_data.strain),
            markersize=5,
            color=ColorSchemes.Dark2_7[i],
        )
    end # for strain_data

    # Check if is the first plot
    if idx == 1
        # Add legend to each subplot
        Legend(gl[:, col+2], axes[row, col], "Strain")
    end # if
end # for

save("$(fig_dir)/rhvae_latent_spaces_strain.pdf", fig)
save("$(fig_dir)/rhvae_latent_spaces_strain.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by day...")

# Initialize figure
fig = Figure(size=(850, 650))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(gl[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Define color palette
unique_days = sort(unique(df_latent.day))
colors = get(ColorSchemes.Blues_9, range(0.5, 1.0, length=length(unique_days)))

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Group dataframe by :day for this model
    df_group = DF.groupby(data, :day)

    # Loop over day groups
    for (i, day_data) in enumerate(df_group)
        # Plot latent space
        scatter!(
            axes[row, col],
            day_data.latent1,
            day_data.latent2,
            label="$(first(day_data.day))",
            markersize=5,
            color=colors[findfirst(==(first(day_data.day)), unique_days)],
        )
    end # for day_data

    # Check if it's the first plot
    if idx == 1
        # Add legend to the right of the grid
        Legend(gl[:, 3], axes[row, col], "Day", nbanks=3)
    end # if
end # for

save("$(fig_dir)/rhvae_latent_spaces_day.pdf", fig)
save("$(fig_dir)/rhvae_latent_spaces_day.png", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by strain number for each model...")

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Initialize figure
fig = Figure(size=(650, 650))

# Create a 2x2 grid of axes
axes = [Axis(fig[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Group dataframe by :strain_num for this model
    df_group = DF.groupby(data, :strain_num)

    # Loop over strain_num groups
    for (i, strain_data) in enumerate(df_group)
        # Plot latent space
        scatter!(
            axes[row, col],
            strain_data.latent1,
            strain_data.latent2,
            markersize=5,
            color=ColorSchemes.glasbey_hv_n256[i],
        )
    end # for strain_data
end # for

save("$(fig_dir)/rhvae_latent_spaces_strain_num.pdf", fig)
save("$(fig_dir)/rhvae_latent_spaces_strain_num.png", fig)

fig

## =============================================================================

println("Compute Riemannian metric for each model...")

# Define number of points per axis
n_points = 150

# Initialize dictionary to store metrics and ranges
metrics = Dict()

# Compute metric for each model
for model_evo in unique(df_latent.model_evo)
    # Filter data for this model
    df_model = df_latent[df_latent.model_evo.==model_evo, :]

    # Extract latent space ranges for this model
    latent1_range = range(
        minimum(df_model.latent1) - 1.5,
        maximum(df_model.latent1) + 1.5,
        length=n_points
    )
    latent2_range = range(
        minimum(df_model.latent2) - 1.5,
        maximum(df_model.latent2) + 1.5,
        length=n_points
    )

    # Define latent points to evaluate
    z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

    # Extract model from dictionary
    rhvae = rhvae_models[model_evo]

    # Compute inverse metric tensor
    Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

    # Compute metric 
    logdetG = reshape(
        -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
    )

    # Store in dictionary
    metrics[model_evo] = Dict(
        "logdetG" => logdetG,
        "latent1" => latent1_range,
        "latent2" => latent2_range
    )
end

## =============================================================================

println("Plotting latent space metric for each model...")

# Initialize figure
fig = Figure(size=(650, 650))

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(fig[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space metric for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        axes[row, col],
        metrics[model]["latent1"],
        metrics[model]["latent2"],
        metrics[model]["logdetG"],
        colormap=ColorSchemes.tokyo,
    )

    # Plot latent space points
    data = df_latent[df_latent.model_evo.==model, :]
    scatter!(
        axes[row, col],
        data.latent1,
        data.latent2,
        markersize=3,
        color=:white,
        alpha=0.5
    )
end

save("$(fig_dir)/rhvae_latent_space_metrics.png", fig)
save("$(fig_dir)/rhvae_latent_space_metrics.pdf", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by strain number...")

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Initialize figure
fig = Figure(size=(650, 650))

# Create a 2x2 grid of axes
axes = [Axis(fig[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        axes[row, col],
        metrics[model]["latent1"],
        metrics[model]["latent2"],
        metrics[model]["logdetG"],
        colormap=ColorSchemes.tokyo,
    )
    # Group dataframe by :strain_num for this model
    df_group = DF.groupby(data, :strain_num)

    # Loop over strain_num groups
    for (i, strain_data) in enumerate(df_group)

        # Plot latent space
        scatter!(
            axes[row, col],
            strain_data.latent1,
            strain_data.latent2,
            markersize=5,
            color=ColorSchemes.glasbey_hv_n256[i],
        )
    end # for strain_data
end # for

save("$(fig_dir)/rhvae_latent_space_metric_strain_num.png", fig)
save("$(fig_dir)/rhvae_latent_space_metric_strain_num.pdf", fig)

fig

## =============================================================================

println("Plotting latent space coordinates colored by training set for each model...")

# Initialize figure
fig = Figure(size=(800, 650))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Define colors
train_color = ColorSchemes.seaborn_colorblind[1]
validation_color = ColorSchemes.seaborn_colorblind[2]

# Create a 2x2 grid of axes
axes = [Axis(gl[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Group data by :train
    data_group = DF.groupby(data, :train)
    # Loop over groups
    for (i, group) in enumerate(data_group[end:-1:1])
        # Plot training set
        scatter!(
            axes[row, col],
            group.latent1,
            group.latent2,
            markersize=5,
            color=first(group.train) ? train_color : validation_color,
            label=first(group.train) ? "training" : "validation",
        )
    end # for
    # Check if is the first plot
    if idx == 1
        # Add legend to each subplot
        Legend(gl[:, col+2], axes[row, col], "Set")
    end # if

end # for

save("$(fig_dir)/rhvae_latent_space_train_val_by_model.png", fig)
save("$(fig_dir)/rhvae_latent_space_train_val_by_model.pdf", fig)

fig

## =============================================================================


println("Plotting latent space coordinates colored by training set with metric...")

# Initialize figure
fig = Figure(size=(800, 650))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Get unique model_evo values
model_evos = unique(df_latent.model_evo)

# Create a 2x2 grid of axes
axes = [Axis(gl[i, j],
    xlabel="latent dimension 1",
    ylabel="latent dimension 2",
    aspect=AxisAspect(1),
    title="Condition removed: $(model_evos[(i-1)*2 + j])")
        for i in 1:2, j in 1:2]

# Plot latent space for each model
for (idx, model) in enumerate(model_evos)
    row, col = divrem(idx - 1, 2) .+ 1
    data = df_latent[df_latent.model_evo.==model, :]

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        axes[row, col],
        latent1_range,
        latent2_range,
        metrics[model],
        colormap=ColorSchemes.tokyo,
    )

    # Group data by :train
    data_group = DF.groupby(data, :train)
    # Loop over groups
    for (i, group) in enumerate(data_group)
        # Plot training set
        scatter!(
            axes[row, col],
            group.latent1,
            group.latent2,
            markersize=5,
            color=ColorSchemes.seaborn_colorblind[i],
            label=first(group.train) ? "training" : "validation",
        )
    end # for
    # Check if is the first plot
    if idx == 1
        # Add legend to each subplot
        Legend(gl[:, col+2], axes[row, col], "Set")
    end # if

end # for

save("$(fig_dir)/rhvae_latent_space_train_val_metric.png", fig)
save("$(fig_dir)/rhvae_latent_space_train_val_metric.pdf", fig)

fig

## =============================================================================