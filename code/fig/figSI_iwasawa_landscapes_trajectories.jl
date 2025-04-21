## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic
import Antibiotic.mh as mh

# Import AutoEncoderToolkit to train VAEs
import AutoEncoderToolkit as AET

# Import libraries to handle data
import Glob
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
import MultivariateStats as MStats
import StatsBase
import Random

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

println("Defining directories...")

# Define model directory
model_dir = "$(git_root())/output/" *
            "beta-rhvae_jointlogencoder_simpledecoder_iwasawa_mcmc/v05"
# Define state directory
rhvae_state_dir = "$(model_dir)/model_state"

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(model_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading IC50 data...")

# Load logic50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load list of drugs
drug_list = sort(unique(df_logic50.drug))
# Load standardize data
data_std = JLD2.load("$(data_dir)/logic50_preprocess.jld2")["logic50_mean_std"]

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
            :model .=> :rhvae
        )
    )
end # for 

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 300

# Extract latent space ranges
latent1_range = range(
    minimum(df_latent[df_latent.model.==:rhvae, :latent1]) - 1.5,
    maximum(df_latent[df_latent.model.==:rhvae, :latent1]) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(df_latent[df_latent.model.==:rhvae, :latent2]) - 1.5,
    maximum(df_latent[df_latent.model.==:rhvae, :latent2]) + 1.5,
    length=n_points
)
# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])

# Compute inverse metric tensor
Ginv = AET.RHVAEs.G_inv(z_mat, rhvae)

# Decode latent points
ic50_rhvae = reshape(
    rhvae.vae.decoder(z_mat).µ',
    n_points,
    n_points,
    length(drug_list)
)

# Compute metric 
logdetG = reshape(
    -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
)

## =============================================================================

# Define mask for fitness landscape
mask = (maximum(logdetG) * 0.92 .< logdetG .<= maximum(logdetG))

## =============================================================================

# Define list of antibiotics
antibiotics = ["KM", "NFLX", "TET"]

# Define azimuth angle for each antibiotic
azimuth = Dict(
    "KM" => 1.275π,
    "NFLX" => 0.25π,
    "TET" => 0.25π
)

# Loop over antibiotics
for antibiotic in antibiotics
    # Initialize figure
    fig = Figure(size=(800, 400))

    # Add global GridLayout to insert axis
    gl = fig[1, 1] = GridLayout()

    # Add grid layout for banner
    gl_banner = gl[1, 1] = GridLayout()

    # Add grid layout for subplots
    gl_subplots = gl[2, 1] = GridLayout()

    # --------------------------------------------------------------------------
    # Add banner
    # --------------------------------------------------------------------------

    # Add box for section title
    Box(
        gl_banner[1, 1],
        color="#E6E6EF",
        strokecolor="#E6E6EF",
        alignmode=Mixed(; left=-40, right=-40), # Moves box to the left and right
    )

    # Add section title
    Label(
        gl_banner[1, 1],
        "trajectories in $(antibiotic) fitness landscape",
        fontsize=12,
        padding=(-30, 0, 0, 0),
        halign=:left,
        tellwidth=false, # prevent column from contracting because of label size
    )

    # --------------------------------------------------------------------------
    # Add subplots
    # --------------------------------------------------------------------------

    # Add 2D axis to plot
    ax1 = Axis(
        gl_subplots[1, 1],
        xlabel="latent dimension 1",
        ylabel="latent dimension 2",
        aspect=AxisAspect(1)
    )

    # Add 3D axis to plot
    ax2 = Axis3(
        gl_subplots[1, 3],
        xlabel="latent dimension 1",
        ylabel="latent dimension 2",
        zlabel="log(IC₅₀)",
        xlabelsize=12,
        ylabelsize=12,
        zlabelsize=12,
        aspect=(1, 1, 0.5),
        azimuth=azimuth[antibiotic],
        elevation=0.25π,
        protrusions=(30, 10, 30, 20),
    )
    # Define fitness landscape
    fit_landscape = dropdims(ic50_rhvae[:, :, drug_list.==antibiotic], dims=3)

    # Mask fitness landscape
    fit_landscape_masked = (mask .* minimum(fit_landscape)) .+
                           (fit_landscape .* .!mask)

    # Plot heatmap of predicted resistance
    hm = heatmap!(
        ax1,
        latent1_range,
        latent2_range,
        fit_landscape_masked,
        colormap=:algae,
        rasterize=true,
    )

    # Add contour lines
    contour!(
        ax1,
        latent1_range,
        latent2_range,
        fit_landscape_masked,
        color=:black,
        linestyle=:dash,
        levels=7,
    )

    # Add colorbar
    cb = Colorbar(
        gl_subplots[1, 2],
        hm,
        size=8,
        label="log(IC₅₀)",
        labelsize=12,
        labelpadding=0.0,
        ticklabelsize=12,
        ticksvisible=false
    )

    # Plot surface of predicted resistance
    surface!(
        ax2,
        latent1_range,
        latent2_range,
        fit_landscape_masked,
        colormap=:algae,
        alpha=0.25,
        rasterize=true,
    )

    # Group data by strain
    df_group = DF.groupby(df_latent[df_latent.env.==antibiotic, :], :strain_num)

    # Loop through groups
    for (j, data) in enumerate(df_group)
        # Sort data by day
        data = DF.sort!(data, :day)
        # Add scatter plot to axis
        scatterlines!(
            ax1,
            data.latent1,
            data.latent2,
            markersize=7,
            label=data.strain[1],
            color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
            linewidth=2
        )
    end

    # Loop through groups
    for (j, data) in enumerate(df_group)
        # Add first point 
        scatter!(
            ax1,
            [data.latent1[1]],
            [data.latent2[1]],
            markersize=18,
            marker=:xcross,
            color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
            strokewidth=2,
            strokecolor=:white,
        )

        # Add last point
        scatter!(
            ax1,
            [data.latent1[end]],
            [data.latent2[end]],
            markersize=18,
            marker=:utriangle,
            color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
            strokewidth=2,
            strokecolor=:white,
        )
    end

    # Define number of points for line
    n_line = 25

    # Loop through groups
    for (j, data) in enumerate(df_group)
        # Sort data by day
        data = DF.sort!(data, :day)
        # Loop through points
        for i in 2:size(data, 1)
            # Extract latent coordinates for this point
            latent_final = [data.latent1[i], data.latent2[i]]
            # Extract latent coordinates for previous point
            latent_initial = [data.latent1[i-1], data.latent2[i-1]]
            # Generate linear interpolation between points
            line = Antibiotic.geometry.linear_interpolation(
                latent_initial,
                latent_final,
                n_line
            )
            # Decode line
            ic50_line = reshape(
                rhvae.vae.decoder(line).µ',
                n_line,
                length(drug_list)
            )
            # Plot line
            lines!(
                ax2,
                line[1, :],
                line[2, :],
                vec(ic50_line[:, drug_list.==antibiotic]),
                color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
                linewidth=2,
            )

            # Add scatter points to initial 
            scatter!(
                ax2,
                Point3f(
                    line[1, end],
                    line[2, end],
                    ic50_line[end, drug_list.==antibiotic][1]
                ),
                color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
                markersize=4,
            )

            # Add initial point
            if i == 2
                # Add initial point
                scatter!(
                    ax2,
                    Point3f(
                        data.latent1[1],
                        data.latent2[1],
                        ic50_line[1, drug_list.==antibiotic][1]
                    ),
                    markersize=10,
                    marker=:xcross,
                    color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
                    strokewidth=2,
                    strokecolor=:white,
                )
            end
            # Add last point    
            if i == size(data, 1)
                # Add last point
                scatter!(
                    ax2,
                    Point3f(
                        data.latent1[end],
                        data.latent2[end],
                        ic50_line[end, drug_list.==antibiotic][1]
                    ),
                    markersize=10,
                    marker=:utriangle,
                    color=ColorSchemes.glasbey_bw_minc_20_hue_330_100_n256[j],
                    strokewidth=2,
                    strokecolor=:white,
                )
            end
        end
    end

    # Save figure
    save("$(fig_dir)/figSI_iwasawa_landscapes_trajectories_$(antibiotic).pdf", fig)
    save("$(fig_dir)/figSI_iwasawa_landscapes_trajectories_$(antibiotic).png", fig)

end # for antibiotic

fig