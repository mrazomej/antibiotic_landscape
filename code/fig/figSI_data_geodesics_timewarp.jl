## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import AutoEncoderToolkit.diffgeo.NeuralGeodesics as NG

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
import DimensionalData as DD
import DataFrames as DF
import CSV

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import IterTools for Cartesian product
import IterTools

# Import basic math
import LinearAlgebra
import StatsBase
import Random

# Import library for dynamic time warping
import DynamicAxisWarping as DAW
import Distances

# Load Plotting packages
using CairoMakie
using Makie
import ColorSchemes
import Colors
# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## =============================================================================

# Define model directory
model_dir = "$(git_root())/output/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/v05"
# Define state directory
state_dir = "$(model_dir)/model_state"
# Define geodesic directory
geodesic_dir = "$(model_dir)/geodesic_state"

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
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
nng_template = JLD2.load("$(model_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Load RHVAE model
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading IC50 data...")

df_logic50 = CSV.File(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_ci.csv"
) |> DF.DataFrame
# Load list of drugs
drug_list = sort(unique(df_logic50.drug))
# Load data matrix
logic50_mat = JLD2.load(
    "$(git_root())/output/mcmc_iwasawa_logistic/logic50_preprocess.jld2")["logic50_mean_std"]

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
# Figure SI_data_geodesics_timewarp
## =============================================================================

# Define (:env, :strain_num) pairs to plot
pairs = [
    ("KM", 16), ("NFLX", 33), ("TET", 8), ("KM", 28), ("NFLX", 35), ("TET", 3)
]

# Initialize figure
fig = Figure(size=(700, 900))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Define number of strains per plot
n_strains = 3

# Add grid layouts as rows for each strain
gl_strains = [gl[i, 1] = GridLayout() for i in 1:n_strains]

# Define number of rows and columns
rows = 2
cols = 4

# Loop through each strain
for i in 1:n_strains
    # Extract pair
    p = pairs[i]

    println("env: $(p[1]) | strain: $(p[2])")
    # --------------------------------------------------------------------------
    # Extract metadata
    data_meta = df_meta[
        (df_meta.env.==p[1]).&(df_meta.strain_num.==p[2]), :
    ]

    # Extract lineage information
    lineage = df_latent[df_latent.strain_num.==p[2], :]

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
    # Run curve through decoder
    curve_decoded = rhvae.vae.decoder(curve).μ
    # Compute time warp
    cost, idx1, idx2 = DAW.dtw(
        Float32.(Matrix(permutedims(lineage[:, [:latent1, :latent2]]))),
        curve,
        transportcost=1
    )

    # --------------------------------------------------------------------------

    # Add grid layout as left column for latent space
    gl_latent_banner = gl_strains[i][1, 1] = GridLayout()
    gl_latent = gl_strains[i][2, 1] = GridLayout()

    # Add grid layout as right column for timewarp
    gl_timewarp_banner = gl_strains[i][1, 2] = GridLayout()
    gl_timewarp = gl_strains[i][2, 2] = GridLayout()

    # --------------------------------------------------------------------------

    # Add banner for latent space
    Box(
        gl_latent_banner[1, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=-10, right=-15),
    )

    # Add section title
    Label(
        gl_latent_banner[1, 1],
        "latent space",
        fontsize=12,
        padding=(-5, 0, 0, 0),
        halign=:left,
        tellwidth=false,
    )

    # Add banner for latent space
    Box(
        gl_timewarp_banner[1, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=-20, right=0),
    )

    # Add section title
    Label(
        gl_timewarp_banner[1, 1],
        "comparison of time-warped predicted curves with experimental data",
        fontsize=12,
        padding=(-5, 0, 0, 0),
        halign=:left,
        tellwidth=false,
    )

    # --------------------------------------------------------------------------

    println("   - plotting latent space")

    # Add axis for latent space
    ax_latent = Axis(
        gl_latent[1, 1],
        aspect=AxisAspect(1),
        xticksvisible=false,
        yticksvisible=false,
        title="$(p[1]) (selection)",
        titlesize=12,
    )
    # Hide axis labels
    hidedecorations!(ax_latent)

    # Plot heatmap of log determinant of metric tensor
    hm = heatmap!(
        ax_latent,
        latent1_range,
        latent2_range,
        logdetG,
        colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
        alpha=0.5,
    )

    # Plot all points in background
    scatter!(
        ax_latent,
        df_latent.latent1,
        df_latent.latent2,
        markersize=5,
        color=(:white, 0.10),
        marker=:circle,
    )

    # Plot lineage
    scatterlines!(
        ax_latent,
        lineage.latent1,
        lineage.latent2,
        linewidth=2,
        color=Antibiotic.viz.colors()[:gold],
        markersize=6,
        label="lineage\nlatent\ncoordinates",
    )

    # Add geodesic line to axis
    lines!(
        ax_latent,
        eachrow(curve)...,
        linewidth=2.5,
        color=Antibiotic.viz.colors()[:dark_red],
        label="geodesic",
    )

    scatter!(
        ax_latent,
        [lineage.latent1[1]],
        [lineage.latent2[1]],
        color=:black,
        markersize=12,
        marker=:xcross,
        label="start",
    )

    scatter!(
        ax_latent,
        [lineage.latent1[end]],
        [lineage.latent2[end]],
        color=:black,
        markersize=12,
        marker=:utriangle,
        label="end",
    )

    # --------------------------------------------------------------------------

    println("   - plotting timewarp")

    # Loop through drugs
    for (j, drug) in enumerate(drug_list)
        # Define row and column index
        row = (j - 1) ÷ cols + 1
        col = (j - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_timewarp[row, col],
            aspect=AxisAspect(1.25),
            title=drug == p[1] ? "$(drug) (selection)" : "$(drug)",
            titlesize=12,
            xticklabelsize=10,
            yticklabelsize=10,
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_meta.strain_num[1]), :]

        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            color=Antibiotic.viz.colors()[:dark_gold],
            linewidth=2.5,
            markersize=6,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
            color=Antibiotic.viz.colors()[:dark_gold],
        )

        # Plot decoded curve
        # Create scaled x-axis from min to max day values
        scaled_days = range(
            minimum(data_strain.day),
            maximum(data_strain.day),
            length=length(curve_decoded[j, :])
        )

        # Plot decoded curve with scaled days
        lines!(
            ax,
            data_strain.day[idx1],
            curve_decoded[j, idx2],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )
    end # for j in 1:length(drug_list)

    # Add axis labels
    Label(
        gl_timewarp[end, :, Bottom()],
        "day",
        fontsize=12,
        padding=(0, 0, 0, 15),
    )

    Label(
        gl_timewarp[:, 1, Left()],
        "log(IC₅₀)",
        fontsize=12,
        rotation=π / 2,
        padding=(0, 0, 0, 0),
    )
    # Adjust layout
    colsize!(gl_strains[i], 1, Auto(1))
    colsize!(gl_strains[i], 2, Auto(5))
end # for i in 1:n_strains


fig