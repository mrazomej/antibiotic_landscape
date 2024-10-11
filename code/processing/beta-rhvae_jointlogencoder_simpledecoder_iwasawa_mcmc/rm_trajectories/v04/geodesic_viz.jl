## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

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

println("Setting up directories...")

# Locate current directory
path_dir = pwd()

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Find the path prefix where to store output
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match, "processing" => ""
)

# Define output directory
out_dir = "$(git_root())/output$(out_prefix)"

# Define model directory
model_dir = "$(git_root())/output$(out_prefix)/model_state"

# Define directory to store trained geodesic curves
geodesic_dir = "$(git_root())/output$(out_prefix)/geodesic_state/"

# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading RHVAE model metadata...")

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
                :model => evo
            )
        )
    end # for rhvae_models
end # for df_group

## =============================================================================

println("Loading geodesic states metadata...")

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    day_init = parse(Int, match(r"dayinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    day_final = parse(Int, match(r"dayfinal(\d+)", gf).captures[1])
    # Extract evo stress number from file name using regular expression
    evo = match(r"evoenv(\w+)_id", gf).captures[1]
    # Extract GRN id from file name using regular expression
    strain_num = parse(Int, match(r"id(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract RHVAE model from file name
    rhvae_model = match(r"_rhvaemodel(\w+)_rhvaeepoch", gf).captures[1]
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :day_init => day_init,
            :day_final => day_final,
            :evo => evo,
            :strain_num => strain_num,
            :rhvae_epoch => rhvae_epoch,
            :rhvae_model => rhvae_model,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

# Sort dataframe by environment
DF.sort!(df_meta, :evo)

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

println("Compute Riemannian metric for each model...")

# Define number of points per axis
n_points = 150

# Initialize dictionary to store metrics and ranges
metrics = Dict()

# Compute metric for each model
for model_evo in unique(df_latent.model)
    # Filter data for this model
    df_model = df_latent[df_latent.model.==model_evo, :]

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

# Group data by :evo
df_group = DF.groupby(df_latent, :evo)

# Define figure name prefix
fname_prefix = "$(fig_dir)/geodesic_latent_trajectory"

# Define colors
train_color = ColorSchemes.seaborn_colorblind[1]
validation_color = ColorSchemes.seaborn_colorblind[2]

# Loop through each RHVAE model
for (model_evo, rhvae) in rhvae_models
    println("Processing model: $model_evo")

    # Loop through meta chunks
    for (i, df_env) in enumerate(df_group)
        # Extract unique strain numbers
        strain_nums = unique(df_env.strain_num)
        # Define number of columns
        cols = 4
        # Define the number of needed rows
        rows = ceil(Int, length(strain_nums) / cols)
        # Initialize figure
        fig = Figure(size=(250 * cols, 250 * rows))
        # Add grid layout
        gl = fig[1, 1] = GridLayout()

        # Initialize counter
        j = 1

        # Loop through each strain number
        for strain_num in strain_nums
            # Extract metadata
            data_meta = df_meta[
                (df_meta.evo.==first(df_env.evo)).&(df_meta.strain_num.==strain_num).&(df_meta.rhvae_model.==model_evo),
                :]

            println("   - Plotting geodesic: $(strain_num)")
            # Define row and column index
            row = (j - 1) ÷ cols + 1
            col = (j - 1) % cols + 1
            # Add axis
            ax = Axis(
                gl[row, col],
                aspect=AxisAspect(1),
                title="evo-condition $(first(data_meta.evo))\nmodel $(first(data_meta.rhvae_model))",
                xticksvisible=false,
                yticksvisible=false,
            )
            # Hide axis labels
            hidedecorations!(ax)

            # Plot heatmap of log determinant of metric tensor
            heatmap!(
                ax,
                metrics[model_evo]["latent1"],
                metrics[model_evo]["latent2"],
                metrics[model_evo]["logdetG"],
                colormap=ColorSchemes.tokyo,
            )

            # Plot all points in background
            scatter!(
                ax,
                df_latent[df_latent.model.==model_evo, :latent1],
                df_latent[df_latent.model.==model_evo, :latent2],
                markersize=5,
                color=(:gray, 0.25),
                marker=:circle,
            )

            # Extract lineage information
            lineage = df_latent[
                (df_latent.evo.==first(df_env.evo)).&(df_latent.strain_num.==strain_num).&(df_latent.model.==model_evo),
                :]

            # Plot lineage
            scatterlines!(
                ax,
                lineage.latent1,
                lineage.latent2,
                markersize=6,
                linewidth=2,
                color=first(lineage.train) ? train_color : validation_color,
            )

            # Load geodesic state
            geo_state = JLD2.load(first(data_meta.geodesic_state))
            # Define NeuralGeodesic model
            nng = NG.NeuralGeodesic(
                nng_template,
                geo_state["latent_init"],
                geo_state["latent_end"],
            )
            # Update model state
            Flux.loadmodel!(nng, geo_state["model_state"])
            # Generate curve
            curve = nng(t_array)
            # Add geodesic line to axis
            lines!(
                ax,
                eachrow(curve)...,
                linewidth=2,
                linestyle=(:dot, :dense),
                color=:white,
            )

            # Add first point 
            scatter!(
                ax,
                [lineage.latent1[1]],
                [lineage.latent2[1]],
                color=:white,
                markersize=11,
                marker=:xcross
            )
            scatter!(
                ax,
                [lineage.latent1[1]],
                [lineage.latent2[1]],
                color=:black,
                markersize=7,
                marker=:xcross
            )

            # Add last point
            scatter!(
                ax,
                [lineage.latent1[end]],
                [lineage.latent2[end]],
                color=:white,
                markersize=11,
                marker=:utriangle
            )
            scatter!(
                ax,
                [lineage.latent1[end]],
                [lineage.latent2[end]],
                color=:black,
                markersize=7,
                marker=:utriangle
            )
            # Update counter
            j += 1
        end # for strain_num in strain_nums

        # Save figure 
        save("$(fname_prefix)_selection$(first(df_env.evo))_model$(model_evo).png", fig)
    end # for (i, df_env) in enumerate(df_group)
end # for (model_evo, rhvae) in rhvae_models
