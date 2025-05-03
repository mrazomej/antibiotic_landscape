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

# Import library for dynamic time warping
import DynamicAxisWarping as DAW
import Distances

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
# Define model state directory
rhvae_state_dir = "$(git_root())/output$(path_prefix)/rhvae_model_state"
# Define model crossvalidation directory
rhvae_crossval_dir = "$(git_root())/output$(path_prefix)/rhvae_crossvalidation_state"
# Define directory to store trained geodesic curves
geodesic_dir = "$(git_root())/output$(path_prefix)/geodesic_state/"

# Define figure directory
fig_dir = "$(git_root())/fig$(path_prefix)"

# Create figure directory if it does not exist
if !isdir(fig_dir)
    println("Creating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_geodesic = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    day_init = parse(Int, match(r"dayinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    day_final = parse(Int, match(r"dayfinal(\d+)", gf).captures[1])
    # Extract evo stress number from file name using regular expression
    drug = match(r"evoenv(\w+)_id", gf).captures[1]
    # Extract GRN id from file name using regular expression
    strain_num = parse(Int, match(r"id(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_geodesic,
        DF.DataFrame(
            :day_init => day_init,
            :day_final => day_final,
            :drug => drug,
            :strain_num => strain_num,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

# Sort dataframe by environment
DF.sort!(df_geodesic, :drug)

## =============================================================================

println("Loading data into memory...")

# Define data directory
data_dir = "$(git_root())/output/mcmc_iwasawa_logistic"

# Load standardized mean data
logic50_mean = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_mean_std"]
# Load data metadata
data_meta = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_meta"]
# Load drug list
drug_list = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["drugs"]
# Load IC50 data
df_logic50 = CSV.read("$(data_dir)/logic50_ci.csv", DF.DataFrame)
# Load column metadata
df_col_meta = JLD2.load(
    "$(data_dir)/logic50_preprocess.jld2"
)["logic50_meta"]

## =============================================================================

println("Loading errors...")

# Define loss function hyper-parameters
ϵ = Float32(1E-3) # Leapfrog step size
K = 10 # Number of leapfrog steps
βₒ = 0.3f0 # Initial temperature for tempering

# Define RHVAE hyper-parameters in a NamedTuple
rhvae_kwargs = (
    K=K,
    ϵ=ϵ,
    βₒ=βₒ,
)

# List model states
model_states = sort(
    Glob.glob("$(rhvae_state_dir)/beta-rhvae_*_epoch*.jld2"[2:end], "/")
)

# List crossvalidation states
crossval_states = sort(
    Glob.glob("$(rhvae_crossval_dir)/beta-rhvae_*_crossval*_epoch*.jld2"[2:end], "/")
)

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over model states
for file in model_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    drug = match(r"beta-rhvae_(.+?)rm_epoch", split(file, "/")[end]).captures[1]
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => file,
        :model_state => file,
        :model_type => "original",
        :train_frac => 0.85,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for model_state

# Loop over crossvalidation states
for file in crossval_states
    # Extract epoch number from file name using regular expression
    epoch = parse(Int, match(r"epoch(\d+)", file).captures[1])
    # Extract drug name from file name using regular expression
    drug = match(r"beta-rhvae_(.+?)rm_crossval", split(file, "/")[end]).captures[1]
    # Extract split fraction from file name using regular expression
    split_frac = parse(Float64, match(r"(\d+\.\d+)split", file).captures[1])
    # Load model_state file
    model_load = JLD2.load(file)
    # Extract loss train and loss val
    loss_train = model_load["loss_train"]
    loss_val = model_load["loss_val"]
    # Extract mse train and mse val
    mse_train = model_load["mse_train"]
    mse_val = model_load["mse_val"]
    # Generate temporary dataframe to store metadata
    df_tmp = DF.DataFrame(
        :epoch => epoch,
        :drug => drug,
        :loss_train => loss_train,
        :loss_val => loss_val,
        :mse_train => mse_train,
        :mse_val => mse_val,
        :model_file => file,
        :model_state => file,
        :model_type => "crossvalidation",
        :train_frac => split_frac,
    )
    # Append temporary dataframe to main dataframe
    global df_meta = DF.vcat(df_meta, df_tmp)
end # for model_state

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(out_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

println("Loading models...")

# Group data by drug
df_group = DF.groupby(df_meta, [:drug])

# Initialize dictionary to store models
rhvae_dict = Dict{String,Any}()

# Loop over drugs
for (i, data) in enumerate(df_group)
    # Extract drug name
    drug = data.drug[1]
    # Extract original model
    data_full = DF.sort(data[data.model_type.=="original", :], [:epoch])
    # Extract crossvalidation model
    data_crossval = DF.sort(
        data[data.model_type.=="crossvalidation", :], [:epoch]
    )
    # Load model   
    rhvae_full = JLD2.load("$(out_dir)/model_$(drug)rm.jld2")["model"]
    # Load model state
    model_state = JLD2.load(data_full.model_state[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae_full, model_state)
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae_full)
    # Load second decoder
    decoder_missing = JLD2.load("$(out_dir)/decoder_missing.jld2")["model"]
    # Build second rhvae
    rhvae_crossval = AET.RHVAEs.RHVAE(
        deepcopy(rhvae_full.vae.encoder) * decoder_missing,
        deepcopy(rhvae_full.metric_chain),
        deepcopy(rhvae_full.centroids_data),
        deepcopy(rhvae_full.T),
        deepcopy(rhvae_full.λ)
    )
    # Load model state
    model_state = JLD2.load(data_crossval.model_state[end])["model_state"]
    # Input parameters to model
    Flux.loadmodel!(rhvae_crossval, model_state)
    # Update metric parameters
    AET.RHVAEs.update_metric!(rhvae_crossval)
    # Load train and validation idx
    full_train_idx = JLD2.load(data_full.model_state[end])["train_idx"]
    full_val_idx = JLD2.load(data_full.model_state[end])["val_idx"]
    crossval_train_idx = JLD2.load(data_crossval.model_state[end])["train_idx"]
    crossval_val_idx = JLD2.load(data_crossval.model_state[end])["val_idx"]
    # Store models
    rhvae_dict[drug] = (
        full=rhvae_full,
        crossval=rhvae_crossval,
        full_train_idx=full_train_idx,
        full_val_idx=full_val_idx,
        crossval_train_idx=crossval_train_idx,
        crossval_val_idx=crossval_val_idx,
    )
end # for drug

## =============================================================================

println("Map data to latent space...")

# Group dataframe by :day, :strain_num, and :env
df_group = DF.groupby(df_logic50, [:day, :strain_num, :env])
# Initialize empty dataframe to store latent coordinates
df_latent = DF.DataFrame()
# Loop over groups
for (j, data) in enumerate(df_group)
    # Sort data by drug
    DF.sort!(data, :drug)
    # Loop through drugs
    for (i, drug) in enumerate(data.drug)
        # Find number of row in df_meta with same :strain_num, :day, and :env
        meta_idx = findfirst(
            (df_col_meta.strain_num .== first(data.strain_num)) .&
            (df_col_meta.day .== first(data.day)) .&
            (df_col_meta.env .== first(data.env))
        )
        # Find whether data is train or validation
        full_type = meta_idx in rhvae_dict[drug].full_train_idx ? "train" : "validation"
        crossval_type = meta_idx in rhvae_dict[drug].crossval_train_idx ? "train" : "validation"
        # Define index of drugs to be used
        env_idx = sort(setdiff(1:length(data.drug), i))
        # Extract data to map to latent space
        logic50_std = Float32.(data.logic50_mean_std)
        # Select data to map to latent space
        logic50_full_std = logic50_std[env_idx, :]
        logic50_crossval_std = logic50_std[i, :]

        # Run logic50_mean_std through encoder
        latent_full = rhvae_dict[drug].full(logic50_full_std; latent=true)
        # Extract latent coordinates
        latent = latent_full.phase_space.z_final
        # Reconstruct logic50_mean_std
        logic50_full_recon = rhvae_dict[drug].full.vae.decoder(latent).µ
        logic50_crossval_recon = rhvae_dict[drug].crossval.vae.decoder(latent).µ
        # Compute reconstruction error
        logic50_full_mse = Flux.mse(logic50_full_std, logic50_full_recon)
        logic50_crossval_mse = Flux.mse(logic50_crossval_std, logic50_crossval_recon)
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
                :drug => drug,
                :mse_full => logic50_full_mse,
                :mse_crossval => logic50_crossval_mse,
                :full_type => full_type,
                :crossval_type => crossval_type,
                :meta_idx => meta_idx,
            )
        )
    end # for drug
end # for data

## =============================================================================

println("Evaluate metric tensor for all drugs...")

# Initialize dictionary to store metric tensors
metric_dict = Dict{String,Any}()

# Define number of points
n_points = 100

# Loop over drugs
for (i, drug) in enumerate(drug_list)
    # Extract data
    data_drug = df_latent[df_latent.drug.==drug, :]
    # Defie latent space ranges
    latent1_range = range(
        minimum(data_drug.latent1) - 2.5,
        maximum(data_drug.latent1) + 2.5,
        length=n_points,
    )
    latent2_range = range(
        minimum(data_drug.latent2) - 2.5,
        maximum(data_drug.latent2) + 2.5,
        length=n_points,
    )
    # Define latent points to evaluate
    z_mat = reduce(hcat, [[x, y] for x in latent1_range, y in latent2_range])
    # Compute inverse metric tensor
    Ginv = AET.RHVAEs.G_inv(z_mat, rhvae_dict[drug].full)

    # Compute metric 
    logdetG = reshape(
        -1 / 2 * AET.utils.slogdet(Ginv), n_points, n_points
    )
    # Store in dictionary
    metric_dict[drug] = (
        latent1_range=latent1_range,
        latent2_range=latent2_range,
        logdetG=logdetG,
    )
end # for drug

## =============================================================================

# Group data by :drug
df_group = DF.groupby(df_geodesic, [:drug])

# Define number of columns
cols = 4

# Loop over drugs
for (i, data) in enumerate(df_group)
    # Define number of needed rows
    rows = ceil(Int, size(data, 1) / cols)
    # Extract drug name
    drug = data.drug[1]
    println("Plotting geodesics for $(drug)...")
    # Initialize figure
    fig = Figure(size=(200 * cols, 200 * rows + 100))
    # Loop over each row of data
    for (j, row) in enumerate(eachrow(data))
        # Define row and column
        row_idx = (j - 1) ÷ cols + 1
        col_idx = (j - 1) % cols + 1
        # Define axis
        ax = Axis(
            fig[row_idx, col_idx],
            aspect=AxisAspect(1),
            title="$(row.drug) | strain #$(row.strain_num)",
        )
        # Hide decorations
        hidedecorations!(ax)
        # Plot heatmpat of log determinant of metric tensor
        hm = heatmap!(
            ax,
            metric_dict[drug].latent1_range,
            metric_dict[drug].latent2_range,
            metric_dict[drug].logdetG,
            colormap=Reverse(to_colormap(ColorSchemes.PuBu)),
        )
        # Extract lineage information
        lineage = df_latent[(df_latent.strain_num.==row.strain_num).&(df_latent.drug.==drug), :]
        # Sort data by day
        lineage = DF.sort(lineage, :day)
        # Plot lineage
        scatterlines!(
            ax,
            lineage.latent1,
            lineage.latent2,
            markersize=8,
            linewidth=2,
            color=Antibiotic.viz.colors()[:gold],
        )

        # Load geodesic state
        geo_state = JLD2.load(row.geodesic_state)
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
            linewidth=3,
            color=Antibiotic.viz.colors()[:dark_red],
        )

        # Add first point 
        scatter!(
            ax,
            [lineage.latent1[1]],
            [lineage.latent2[1]],
            color=:black,
            markersize=12,
            marker=:xcross
        )
        # Add last point
        scatter!(
            ax,
            [lineage.latent1[end]],
            [lineage.latent2[end]],
            color=:black,
            markersize=12,
            marker=:utriangle
        )
    end # for row

    # Save figure
    save("$(fig_dir)/geodesic_$(first(data.drug))rm.png", fig)

end # for data

## =============================================================================

# Define filename
fname = "$(fig_dir)/resistance_trajectories_prediction.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by drug
df_group_drug = DF.groupby(df_geodesic, [:drug])

# Define numebr of columns
n_cols = 4

# Loop over drugs
for (data_geodesic_drug) in df_group_drug
    # Extract drug name
    drug = data_geodesic_drug.drug[1]
    # Define needed number of rows
    n_rows = ceil(Int, size(data_geodesic_drug, 1) / n_cols)
    # Initialize figure
    fig = Figure(size=(250 * n_cols, 200 * n_rows + 100))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()
    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()
    # Group data by strain_num
    data_group = DF.groupby(data_geodesic_drug, [:strain_num])
    # Loop over data_group
    for (j, data_geo_strain) in enumerate(data_group)
        # Extract strain number
        strain_num = data_geo_strain.strain_num[1]
        # Define row and column
        row = (j - 1) ÷ n_cols + 1
        col = (j - 1) % n_cols + 1
        # Define axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug) | strain #$(strain_num)",
            xlabel="day",
            ylabel="log2(IC50) in $(drug)",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_geo_strain.strain_num[1]), :]
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
        # Extract geodesic metadata
        geo_state = JLD2.load(first(data_geo_strain.geodesic_state))
        # Define NeuralGeodesic model
        nng = NG.NeuralGeodesic(
            nng_template,
            geo_state["latent_init"],
            geo_state["latent_end"],
        )
        # Generate curve
        curve = nng(t_array)
        # Extract decoder
        decoder = rhvae_dict[drug].crossval.vae.decoder
        # Run curve through decoder
        curve_decoded = decoder(curve).μ
        # Define linear interpolation between beginning and end of curve
        line = Antibiotic.geometry.linear_interpolation(
            geo_state["latent_init"],
            geo_state["latent_end"],
            length(t_array)
        )
        # Run line through decoder
        line_decoded = decoder(line).μ

        # Create scaled x-axis from min to max day values
        scaled_days = range(
            minimum(data_strain.day),
            maximum(data_strain.day),
            length=length(vec(curve_decoded)),
        )

        # Plot decoded curve
        lines!(
            ax,
            scaled_days,
            vec(curve_decoded),
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )
        # Plot decoded line
        lines!(
            ax,
            scaled_days,
            vec(line_decoded),
            color=Antibiotic.viz.colors()[:gold],
            label="linear interpolation",
            linewidth=2.5,
        )
        # Add legend if first plot
        if j == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if j == 1

    end # for data_group_strain

    # Add title
    Label(
        gl[1, 1, Top()],
        "RHVAE train without $(drug)",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -10)
    # Change spacing between legend and subplots
    rowgap!(gl, 0)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end # for data_geodesic_drug

## =============================================================================

println("Plotting time warp trajectory predictions...")

# Define filename
fname = "$(fig_dir)/resistance_trajectories_prediction_timewarp.pdf"

# If file exists, delete it
if isfile(fname)
    rm(fname)
end

# Group df_meta by drug
df_group_drug = DF.groupby(df_geodesic, [:drug])

# Define numebr of columns
n_cols = 4

# Loop over drugs
for (data_geodesic_drug) in df_group_drug
    # Extract drug name
    drug = data_geodesic_drug.drug[1]
    # Define needed number of rows
    n_rows = ceil(Int, size(data_geodesic_drug, 1) / n_cols)
    # Initialize figure
    fig = Figure(size=(250 * n_cols, 200 * n_rows + 100))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()
    # Add grid layour for subplots
    gl_plots = gl[1, 1] = GridLayout()
    # Add grid layout for legend
    gl_legend = gl[2, 1] = GridLayout()
    # Group data by strain_num
    data_group = DF.groupby(data_geodesic_drug, [:strain_num])
    # Loop over data_group
    for (j, data_geo_strain) in enumerate(data_group)
        # Extract strain number
        strain_num = data_geo_strain.strain_num[1]
        # Define row and column
        row = (j - 1) ÷ n_cols + 1
        col = (j - 1) % n_cols + 1
        # Define axis
        ax = Axis(
            gl_plots[row, col],
            aspect=AxisAspect(1.25),
            title="$(drug) | strain #$(strain_num)",
            xlabel="day",
            ylabel="log2(IC50) in $(drug)",
        )
        # Extract strain information
        data_strain = df_logic50[
            (df_logic50.drug.==drug).&(df_logic50.strain_num.==data_geo_strain.strain_num[1]), :]
        # Extract latent information
        data_strain_latent = df_latent[
            (df_latent.drug.==drug).&(df_latent.strain_num.==data_geo_strain.strain_num[1]), :]
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
        # Extract geodesic metadata
        geo_state = JLD2.load(first(data_geo_strain.geodesic_state))
        # Define NeuralGeodesic model
        nng = NG.NeuralGeodesic(
            nng_template,
            geo_state["latent_init"],
            geo_state["latent_end"],
        )
        # Generate curve
        curve = nng(t_array)
        # Extract decoder
        decoder = rhvae_dict[drug].crossval.vae.decoder
        # Run curve through decoder
        curve_decoded = decoder(curve).μ
        # Define linear interpolation between beginning and end of curve
        line = Antibiotic.geometry.linear_interpolation(
            geo_state["latent_init"],
            geo_state["latent_end"],
            length(t_array)
        )
        # Run line through decoder
        line_decoded = decoder(line).μ

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

        # Plot decoded curve
        lines!(
            ax,
            data_strain.day[idx1_geo],
            vec(curve_decoded)[idx2_geo],
            color=Antibiotic.viz.colors()[:dark_red],
            label="RHVAE geodesic",
            linewidth=2.5,
        )
        # Plot decoded line
        lines!(
            ax,
            data_strain.day[idx1_line],
            vec(line_decoded)[idx2_line],
            color=Antibiotic.viz.colors()[:gold],
            label="linear interpolation",
            linewidth=2.5,
        )
        # Add legend if first plot
        if j == 1
            Legend(
                gl_legend[1, 1],
                ax,
                orientation=:horizontal,
                merge=true,
                framevisible=false,
            )
        end # if j == 1

    end # for data_group_strain

    # Add title
    Label(
        gl[1, 1, Top()],
        "RHVAE train without $(drug)",
        fontsize=20,
        padding=(0, 0, 30, 0),
    )

    # Change spacing between subplots
    rowgap!(gl_plots, -10)
    # Change spacing between legend and subplots
    rowgap!(gl, 0)

    # Save figure
    save("tmp.pdf", fig)
    # Append to final figure
    append_pdf!(fname, "tmp.pdf", cleanup=true)
end # for data_geodesic_drug