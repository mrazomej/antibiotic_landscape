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
    env = match(r"evoenv(\w+)_id", gf).captures[1]
    # Extract GRN id from file name using regular expression
    strain_num = parse(Int, match(r"id(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :day_init => day_init,
            :day_final => day_final,
            :env => env,
            :strain_num => strain_num,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

# Sort dataframe by environment
DF.sort!(df_meta, :env)

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Load RHVAE model
rhvae = JLD2.load("$(out_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(model_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)
# Load training and validation indexes
idx_train = JLD2.load("$(out_dir)/model.jld2")["idx_keep"]
idx_val = JLD2.load("$(out_dir)/model.jld2")["idx_rm"]

## =============================================================================

println("Loading IC50 data...")

df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for (i, data) in enumerate(df_group)
    # Check if data is in training set
    if idx_train[i]
        train = true
    else
        train = false
    end
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
            :train => train,
        )
    )
end # for 

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent.latent1) - 1.5,
    maximum(df_latent.latent1) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent.latent2) - 1.5,
    maximum(df_latent.latent2) + 1.5,
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

# Group data by :env
df_group = DF.groupby(df_latent, :env)

# Define figure name prefix
fname_prefix = "$(fig_dir)/geodesic_latent_trajectory"

# Define colors
train_color = ColorSchemes.seaborn_colorblind[1]
validation_color = ColorSchemes.seaborn_colorblind[2]

# Loop through meta chunks
for (i, df_env) in enumerate(df_group)
    # Extract unique strain numbers
    strain_nums = unique(df_env.strain_num)
    # Define number of columns
    cols = 4
    # Define the number of needed rows
    rows = ceil(Int, length(strain_nums) / cols)
    # Initialize figure
    fig = Figure(size=(200 * cols, 200 * rows))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Group data by :train column
    data_group = DF.groupby(df_env, :train)

    # Initialize counter
    j = 1

    # Loop through each group
    for train_group in Iterators.reverse(data_group)
        # Check if data is in training set
        is_train = first(train_group.train)
        # Extract unique strain numbers
        strain_nums = unique(train_group.strain_num)
        # Loop through metadata
        for (k, strain_num) in enumerate(strain_nums)
            # Extract metadata
            data_meta = df_meta[
                (df_meta.env.==first(train_group.env)).&(df_meta.strain_num.==strain_num),
                :]

            println("   - Plotting geodesic: $(strain_num) ($(is_train ? "train" : "validation"))")
            # Define row and column index
            row = (j - 1) ÷ cols + 1
            col = (j - 1) % cols + 1
            # Add axis
            ax = Axis(
                gl[row, col],
                aspect=AxisAspect(1),
                title="$(first(data_meta.env)) | " *
                      "$(is_train ? "train" : "validation")",
                xticksvisible=false,
                yticksvisible=false,
            )
            # Hide axis labels
            hidedecorations!(ax)

            # Plot heatmap of log determinant of metric tensor
            heatmap!(
                ax,
                latent1_range,
                latent2_range,
                logdetG,
                colormap=ColorSchemes.tokyo,
            )

            # Plot all points in background
            scatter!(
                ax,
                df_latent.latent1,
                df_latent.latent2,
                markersize=5,
                color=(:gray, 0.25),
                marker=:circle,
            )

            # Extract lineage information
            lineage = train_group[train_group.strain_num.==strain_num, :]

            # Plot lineage
            scatterlines!(
                ax,
                lineage.latent1,
                lineage.latent2,
                markersize=6,
                linewidth=2,
                color=is_train ? train_color : validation_color,
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
        save("$(fname_prefix)_$(first(train_group.env))_train-validation.png", fig)
    end # for (is_train, train_group) in DF.groupby(data_group, :train)
end # for (i, data_group) in enumerate(df_group)
