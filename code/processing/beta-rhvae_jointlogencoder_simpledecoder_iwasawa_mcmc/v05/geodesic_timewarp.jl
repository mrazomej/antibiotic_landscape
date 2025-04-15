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

# Import library for dynamic time warping
import DynamicAxisWarping as DAW
import Distances

# Import basic math
import LinearAlgebra
import MultivariateStats as MStats
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
rhvae_state_dir = "$(git_root())/output$(out_prefix)/model_state"
vae_state_dir = "$(git_root())/output$(out_prefix)/vae_model_state"

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

println("Loading geodesic metadata...")

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

println("Loading RHVAE model...")

# Load RHVAE model
rhvae = JLD2.load("$(out_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(rhvae_state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading vanilla VAE model...")

# Load vanilla VAE model
vae = JLD2.load("$(out_dir)/vae_model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(vae_state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(vae, JLD2.load(param_files[end])["model_state"])

## =============================================================================

println("Loading IC50 data...")

# Load logic50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load list of drugs
drug_list = sort(unique(df_logic50.drug))
# Load data matrix
logic50_mat = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2")["logic50_mean_std"]

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])

# Fit PCA
fit_pca = MStats.fit(MStats.PCA, logic50_mat, maxoutdim=2)

# Initialize empty dataframe to store latent coordinates
df_rhvae = DF.DataFrame()
df_vae = DF.DataFrame()
df_pca = DF.DataFrame()
# Loop over groups
for data in df_group
    # Sort data by drug
    DF.sort!(data, :drug)
    # Run :logic50_mean_std through RHVAE encoder
    latent_rhvae = rhvae.vae.encoder(data.logic50_mean_std).µ
    # Run :logic50_mean_std through VAE encoder
    latent_vae = vae.encoder(data.logic50_mean_std).µ
    # Run :logic50_mean_std through PCA
    latent_pca = MStats.predict(fit_pca, data.logic50_mean_std)
    # Append latent coordinates to rhvae dataframe
    DF.append!(
        df_rhvae,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent_rhvae[1, :],
            :latent2 => latent_rhvae[2, :],
            :model => "rhvae",
        )
    )
    # Append latent coordinates to vae dataframe
    DF.append!(
        df_vae,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent_vae[1, :],
            :latent2 => latent_vae[2, :],
            :model => "vae",
        )
    )
    # Append latent coordinates to pca dataframe
    DF.append!(
        df_pca,
        DF.DataFrame(
            :day .=> first(data.day),
            :strain_num .=> first(data.strain_num),
            :meta .=> first(data.env),
            :env .=> split(first(data.env), "_")[end],
            :strain .=> split(first(data.env), "_")[1],
            :latent1 => latent_pca[1, :],
            :latent2 => latent_pca[2, :],
            :model => "pca",
        )
    )
end # for data in df_group

# Append rhvae and vae dataframes
df_latent = vcat(df_rhvae, df_vae, df_pca)

## =============================================================================

println("Plotting resistance trajectories with timewarp...")

# Define filename
fname = "$(fig_dir)/resistance_trajectories_timewarp.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Loop through groups
for data_meta in df_group
    println("Plotting resistance trajectory: $(data_meta.env[1]) | strain #$(data_meta.strain_num[1])")
    # Initialize figure
    fig = Figure(size=(900, 500))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4

    # Extract geodesic metadata
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

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug)",
            xlabel="days",
            ylabel="logIC₅₀",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_meta.strain_num[1]), :]
        # Compute time warp
        cost, idx1, idx2 = DAW.dtw(
            Float32.(data_strain.logic50_mean_std),
            curve_decoded[i, :],
            transportcost=1
        )

        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            linewidth=2.5,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
        )

        # Plot decoded curve
        # Create scaled x-axis from min to max day values
        scaled_days = range(
            minimum(data_strain.day),
            maximum(data_strain.day),
            length=length(curve_decoded[i, :])
        )

        # Plot decoded curve with scaled days
        lines!(
            ax,
            data_strain.day[idx1],
            curve_decoded[i, idx2],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )

        # Add legend if first plot
        if i == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if i == 1
    end # for drug in drug_list

    # Add title
    Label(
        gl[1, 1, Top()],
        "$(data_meta.env[1]) | strain #$(data_meta.strain_num[1])",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -20)

    # Change spacing between legend and subplots
    rowgap!(gl, -10)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end

## =============================================================================

println("Plotting resistance trajectories with latent space timewarp...")

# Define filename
fname = "$(fig_dir)/resistance_trajectories_timewarp_latent.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Loop through groups
for data_meta in df_group
    println("Plotting resistance trajectory: $(data_meta.env[1]) | strain #$(data_meta.strain_num[1])")
    # Initialize figure
    fig = Figure(size=(900, 550))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4

    # Extract strain latent space information
    data_strain_rhvae = DF.sort(
        df_latent[
            (df_latent.strain_num.==data_meta.strain_num[1]).&(df_latent.model.=="rhvae"), :],
        [:day]
    )
    data_strain_vae = DF.sort(
        df_latent[
            (df_latent.strain_num.==data_meta.strain_num[1]).&(df_latent.model.=="vae"), :],
        [:day]
    )
    data_strain_pca = DF.sort(
        df_latent[
            (df_latent.strain_num.==data_meta.strain_num[1]).&(df_latent.model.=="pca"), :],
        [:day]
    )

    # Extract geodesic metadata
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

    # Define linear interpolation between beginning and end for RHVAE
    line_rhvae = Float32.(Antibiotic.geometry.linear_interpolation(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_rhvae = rhvae.vae.decoder(line_rhvae).μ

    # Define linear interpolation between beginning and end for VAE
    line_vae = Float32.(Antibiotic.geometry.linear_interpolation(
        [data_strain_vae[1, :latent1], data_strain_vae[1, :latent2]],
        [data_strain_vae[end, :latent1], data_strain_vae[end, :latent2]],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_vae = vae.decoder(line_vae).μ

    # Define linear interpolation between beginning and end for PCA
    line_pca = Float32.(Antibiotic.geometry.linear_interpolation(
        [data_strain_pca[1, :latent1], data_strain_pca[1, :latent2]],
        [data_strain_pca[end, :latent1], data_strain_pca[end, :latent2]],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_pca = MStats.reconstruct(fit_pca, line_pca)

    # Compute time warp
    cost_geo_rhvae, idx1_geo_rhvae, idx2_geo_rhvae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_rhvae[:, [:latent1, :latent2]]))),
        curve,
        transportcost=1
    )
    cost_line_rhvae, idx1_line_rhvae, idx2_line_rhvae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_rhvae[:, [:latent1, :latent2]]))),
        line_rhvae,
        transportcost=1
    )
    cost_line_vae, idx1_line_vae, idx2_line_vae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_vae[:, [:latent1, :latent2]]))),
        line_vae,
        transportcost=1
    )
    cost_line_pca, idx1_line_pca, idx2_line_pca = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_pca[:, [:latent1, :latent2]]))),
        line_pca,
        transportcost=1
    )

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug)",
            xlabel="days",
            ylabel="logIC₅₀",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_meta.strain_num[1]), :]

        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            linewidth=2.5,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
        )

        # Plot PCA line
        lines!(
            ax,
            data_strain.day[idx1_line_pca],
            line_decoded_pca[i, idx2_line_pca],
            color=Antibiotic.viz.colors()[:gold],
            label="PCA linear interpolation",
            linewidth=2.5,
        )
        # Plot VAE line
        lines!(
            ax,
            data_strain.day[idx1_line_vae],
            line_decoded_vae[i, idx2_line_vae],
            color=Antibiotic.viz.colors()[:dark_green],
            label="VAE linear interpolation",
            linewidth=2.5,
        )
        # Plot RHVAE line
        lines!(
            ax,
            data_strain.day[idx1_line_rhvae],
            line_decoded_rhvae[i, idx2_line_rhvae],
            color=Antibiotic.viz.colors()[:purple],
            label="RHVAE linear interpolation",
            linewidth=2.5,
        )
        # Plot decoded curve with scaled days
        lines!(
            ax,
            data_strain.day[idx1_geo_rhvae],
            curve_decoded[i, idx2_geo_rhvae],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )
        # Add legend if first plot
        if i == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
                nbanks=2,
            )
        end # if i == 1
    end # for drug in drug_list

    # Add title
    Label(
        gl[1, 1, Top()],
        "$(data_meta.env[1]) | strain #$(data_meta.strain_num[1])",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -20)

    # Change spacing between legend and subplots
    rowgap!(gl, -10)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end

## =============================================================================

println("Computing timewarp reconstruction error results...")

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Initialize empty list to store results
df_timewarp = []

# Loop through groups
for data_meta in df_group
    # Extract environment and strain number
    env = data_meta.env[1]
    strain_num = data_meta.strain_num[1]

    # Extract strain latent space information
    data_strain_rhvae = DF.sort(
        df_latent[
            (df_latent.strain_num.==strain_num).&(df_latent.model.=="rhvae"), :],
        [:day]
    )
    data_strain_vae = DF.sort(
        df_latent[
            (df_latent.strain_num.==strain_num).&(df_latent.model.=="vae"), :],
        [:day]
    )
    data_strain_pca = DF.sort(
        df_latent[
            (df_latent.strain_num.==strain_num).&(df_latent.model.=="pca"), :],
        [:day]
    )

    # Extract geodesic metadata
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

    # Define linear interpolation between beginning and end for RHVAE
    line_rhvae = Float32.(Antibiotic.geometry.linear_interpolation(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_rhvae = rhvae.vae.decoder(line_rhvae).μ

    # Define linear interpolation between beginning and end for VAE
    line_vae = Float32.(Antibiotic.geometry.linear_interpolation(
        [data_strain_vae[1, :latent1], data_strain_vae[1, :latent2]],
        [data_strain_vae[end, :latent1], data_strain_vae[end, :latent2]],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_vae = vae.decoder(line_vae).μ

    # Define linear interpolation between beginning and end for PCA
    line_pca = Float32.(Antibiotic.geometry.linear_interpolation(
        [data_strain_pca[1, :latent1], data_strain_pca[1, :latent2]],
        [data_strain_pca[end, :latent1], data_strain_pca[end, :latent2]],
        length(t_array)
    ))
    # Run line through decoder
    line_decoded_pca = MStats.reconstruct(fit_pca, line_pca)

    # Compute time warp
    cost_geo_rhvae, idx1_geo_rhvae, idx2_geo_rhvae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_rhvae[:, [:latent1, :latent2]]))),
        curve,
        transportcost=1
    )
    cost_line_rhvae, idx1_line_rhvae, idx2_line_rhvae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_rhvae[:, [:latent1, :latent2]]))),
        line_rhvae,
        transportcost=1
    )
    cost_line_vae, idx1_line_vae, idx2_line_vae = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_vae[:, [:latent1, :latent2]]))),
        line_vae,
        transportcost=1
    )
    cost_line_pca, idx1_line_pca, idx2_line_pca = DAW.dtw(
        Float32.(Matrix(permutedims(data_strain_pca[:, [:latent1, :latent2]]))),
        line_pca,
        transportcost=1
    )

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==strain_num), :]

        # Append RHVAE results to list
        push!(df_timewarp, Dict(
            :env => env,
            :strain_num => strain_num,
            :drug => drug,
            :cost => cost_geo_rhvae,
            :model => "rhvae",
            :type => "geodesic",
            :mse => Flux.mse(
                curve_decoded[i, idx2_geo_rhvae],
                data_strain.logic50_mean_std[idx1_geo_rhvae]
            ),
        ))

        push!(df_timewarp, Dict(
            :env => env,
            :strain_num => strain_num,
            :drug => drug,
            :cost => cost_line_rhvae,
            :model => "rhvae",
            :type => "linear",
            :mse => Flux.mse(
                line_decoded_rhvae[i, idx2_line_rhvae],
                data_strain.logic50_mean_std[idx1_line_rhvae]
            ),
        ))

        # Append VAE results to list
        push!(df_timewarp, Dict(
            :env => env,
            :strain_num => strain_num,
            :drug => drug,
            :cost => cost_line_vae,
            :model => "vae",
            :type => "linear",
            :mse => Flux.mse(
                line_decoded_vae[i, idx2_line_vae],
                data_strain.logic50_mean_std[idx1_line_vae]
            ),
        ))

        # Append PCA results to list
        push!(df_timewarp, Dict(
            :env => env,
            :strain_num => strain_num,
            :drug => drug,
            :cost => cost_line_pca,
            :model => "pca",
            :type => "linear",
            :mse => Flux.mse(
                line_decoded_pca[i, idx2_line_pca],
                data_strain.logic50_mean_std[idx1_line_pca]
            ),
        ))
    end # for drug in drug_list
end

# Convert list to data frame
df_timewarp = DF.DataFrame(df_timewarp)
## =============================================================================

println("Plotting timewarp reconstruction error results...")

# Define colors for each model
colors = Dict(
    "pca" => Antibiotic.viz.colors()[:gold],
    "vae" => Antibiotic.viz.colors()[:dark_green],
    "rhvae" => Antibiotic.viz.colors()[:light_red],
)

# Initialize figure
fig = Figure(size=(900, 500))

# Define number of rows and columns
rows = 2
cols = 4

# Add grid layout
gl = fig[1, 1] = GridLayout()
# Add grid layout for subplots
gl_plots = gl[1, 1] = GridLayout()
# Add grid layout for legend
gl_legend = gl[2, 1] = GridLayout()

# Loop through drugs
for (i, drug) in enumerate(drug_list)
    # Define row and column index
    row = (i - 1) ÷ cols + 1
    col = (i - 1) % cols + 1
    # Add axis
    ax = Axis(
        gl_plots[row, col],
        title="$(drug)",
        aspect=AxisAspect(1.25),
        xlabel="mean squared error",
        ylabel="ECDF",
        xlabelsize=12,
        ylabelsize=12,
    )
    # Loop through models
    for model in ["pca", "vae", "rhvae"]
        # Extract data
        data = df_timewarp[
            (df_timewarp.drug.==drug).&(df_timewarp.model.==model).&(df_timewarp.type.=="linear"), :]
        # Plot ECDF
        ecdfplot!(
            ax,
            data.mse,
            color=colors[model],
            label="$(uppercase(model)) linear interpolation",
            linewidth=2.5,
        )
    end
    # Extract data
    data = df_timewarp[
        (df_timewarp.drug.==drug).&(df_timewarp.type.=="geodesic"), :]
    # Plot ECDF
    ecdfplot!(
        ax,
        data.mse,
        color=Antibiotic.viz.colors()[:dark_red],
        label="RHVAE geodesic",
        linewidth=2.5,
    )
    # Add legend
    if i == 1
        Legend(
            gl_legend[1, 1],
            ax,
            orientation=:horizontal,
            merge=true,
            framevisible=false,
        )
    end
end

# Add title
Label(
    gl[1, 1, Top()],
    "Timewarp reconstruction error",
    fontsize=16,
    padding=(0, 0, 30, 0),
)

# Save figure
save("$(fig_dir)/timewarp_reconstruction_error.pdf", fig)

fig

## =============================================================================

# Initialize figure
fig = Figure(size=(350, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time warping cost",
    ylabel="ECDF",
)

# Loop through models
for model in ["pca", "vae", "rhvae"]
    # Extract data
    data = df_timewarp[
        (df_timewarp.drug.==drug_list[1]).&(df_timewarp.model.==model).&(df_timewarp.type.=="linear"), :]
    # Plot ECDF
    ecdfplot!(
        ax,
        data.cost,
        color=colors[model],
        label="$(uppercase(model))",
        linewidth=2.5,
    )
end

# Extract data
data = df_timewarp[
    (df_timewarp.drug.==drug_list[1]).&(df_timewarp.type.=="geodesic"), :]
# Plot ECDF
ecdfplot!(
    ax,
    data.cost,
    color=Antibiotic.viz.colors()[:dark_red],
    label="RHVAE geodesic",
    linewidth=2.5,
)

# Add legend
axislegend(ax, merge=true, framevisible=false, position=:rb)

# Save figure
save("$(fig_dir)/timewarp_cost_ecdf.pdf", fig)

fig
## =============================================================================

println("Plotting resistance trajectories with latent space timewarp and Brownian bridge...")

# Define filename
fname = "$(fig_dir)/resistance_trajectories_timewarp_latent_brownian.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Define number of Brownian bridges
n_bridges = 100
# Define sigma
sigma = 2.0

# Loop through groups
for data_meta in df_group
    println("Plotting resistance trajectory: $(data_meta.env[1]) | strain #$(data_meta.strain_num[1])")
    # Initialize figure
    fig = Figure(size=(900, 500))

    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()

    # Define number of rows and columns
    rows = 2
    cols = 4

    # Extract geodesic metadata
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

    # Define linear interpolation between beginning and end of curve
    line = Antibiotic.geometry.linear_interpolation(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array)
    )
    # Run line through decoder
    line_decoded = rhvae.vae.decoder(line).μ
    # Generate Brownian bridge
    rnd_bridge = Antibiotic.geometry.brownian_bridge(
        geo_state["latent_init"],
        geo_state["latent_end"],
        length(t_array),
        sigma=sigma,
        num_paths=n_bridges,
        rng=Random.MersenneTwister(42)
    )
    # Decode Brownian bridge
    bridge_decoded = rhvae.vae.decoder(rnd_bridge).μ

    # Extract strain information
    data_strain_latent = df_latent[
        (df_latent.strain_num.==data_meta.strain_num[1]), :]
    # Compute time warp
    cost_geo, idx1_geo, idx2_geo = DAW.dtw(
        Matrix(data_strain_latent[:, [:latent1, :latent2]])',
        curve,
        transportcost=1
    )
    cost_line, idx1_line, idx2_line = DAW.dtw(
        Matrix(data_strain_latent[:, [:latent1, :latent2]])',
        line,
        transportcost=1
    )
    # Compute time warp for each Brownian bridge
    idx1_bridge = []
    idx2_bridge = []
    for i in 1:n_bridges
        _, idx1, idx2 = DAW.dtw(
            Matrix(data_strain_latent[:, [:latent1, :latent2]])',
            rnd_bridge[:, :, i],
            transportcost=1
        )
        push!(idx1_bridge, idx1)
        push!(idx2_bridge, idx2)
    end # for i in 1:n_bridges

    # Loop through drugs
    for (i, drug) in enumerate(drug_list)
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug)",
            xlabel="days",
            ylabel="logIC₅₀",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_meta.strain_num[1]), :]

        # Plot each Brownian bridge
        for j in 1:n_bridges
            lines!(
                ax,
                data_strain.day[idx1_bridge[j]],
                bridge_decoded[i, idx2_bridge[j], j],
                color=(:gray, 0.25),
                linewidth=1.5,
            )
        end # for j in 1:n_bridges

        # Plot strain information
        scatterlines!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            linewidth=2.5,
        )
        # Add error bars
        errorbars!(
            ax,
            data_strain.day,
            data_strain.logic50_mean_std,
            data_strain.logic50_mean_std .- data_strain.logic50_ci_lower_std,
            data_strain.logic50_ci_upper_std .- data_strain.logic50_mean_std,
            label="experimental",
        )

        # Plot decoded curve with scaled days
        lines!(
            ax,
            data_strain.day[idx1_geo],
            curve_decoded[i, idx2_geo],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )

        # Plot line
        lines!(
            ax,
            data_strain.day[idx1_line],
            line_decoded[i, idx2_line],
            color=Antibiotic.viz.colors()[:gold],
            label="linear interpolation",
            linewidth=2.5,
        )

        # Add legend if first plot
        if i == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if i == 1
    end # for drug in drug_list

    # Add title
    Label(
        gl[1, 1, Top()],
        "$(data_meta.env[1]) | strain #$(data_meta.strain_num[1])",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -20)

    # Change spacing between legend and subplots
    rowgap!(gl, -10)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end # for data_meta in df_group

## =============================================================================

# Group df_meta by env and strain_num
df_group = DF.groupby(df_meta, [:env, :strain_num])

# Select first group (for now)
data_meta = first(df_group)

# Extract geodesic metadata
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

# Define linear interpolation between beginning and end of curve
line = Antibiotic.geometry.linear_interpolation(
    geo_state["latent_init"],
    geo_state["latent_end"],
    length(t_array)
)
# Run line through decoder
line_decoded = rhvae.vae.decoder(line).μ

# Extract strain information
data_strain = df_latent[
    (df_latent.env.==first(data_meta.env)).&(df_latent.strain_num.==data_meta.strain_num[1]), :]

cost, idx1, idx2 = DAW.dtw(
    Matrix(data_strain[:, [:latent1, :latent2]])',
    curve,
    transportcost=1
)

rnd_bridge = Antibiotic.geometry.brownian_bridge(
    geo_state["latent_init"],
    geo_state["latent_end"],
    length(t_array),
    sigma=0.1,
    num_paths=25,
    rng=Random.MersenneTwister(42)
)

# Initialize figure
fig = Figure(size=(300, 300))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add axis
ax = Axis(gl[1, 1])

# Loop over Brownian bridges
for (i, bridge) in enumerate(eachslice(rnd_bridge, dims=3))
    lines!(
        ax,
        bridge[1, :],
        bridge[2, :],
        color=(ColorSchemes.glasbey_hv_n256[i], 0.25),
        linewidth=1.5,
    )
end

# Plot geodesic
lines!(ax, curve[1, :], curve[2, :], color=Antibiotic.viz.colors()[:dark_red])

# Plot line
lines!(ax, line[1, :], line[2, :], color=Antibiotic.viz.colors()[:gold])


fig