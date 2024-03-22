## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import package for VAEs
import AutoEncode
import AutoEncode.diffgeo.NeuralGeodesics as NG

# Import library to handle data
import Glob
import CSV
import DataFrames as DF

# Import ML libraries
import Flux

# Import library to save models
import JLD2

# Import Plotting libraries
using CairoMakie
import ColorSchemes
import PDFmerger

# Activate backend
CairoMakie.activate!()

# Set Plotting style
Antibiotic.viz.theme_makie!()

# Import basic math
import Random
import StatsBase

Random.seed!(42)

## =============================================================================

# Define number of time points to evaluate along curve
n_time = 100

## =============================================================================

println("Loading NeuralGeodesic model...")

# Load model template
nng_template = JLD2.load("./output/geodesic.jld2")["model"]

# List all parameter files
nng_files = Glob.glob("./output/geodesic_state/*.jld2")

## =============================================================================

println("Loading RHVAE models...")

# List files in output directory
files = Glob.glob("output/model_state/*.jld2")

# Locate unique temperatures in file names
temps = sort(unique([split(f, "_")[end-1] for f in files]))

# Initialize empty dictionary to store models
rhvae_dict = Dict()

# Load model template
rhvae_template = JLD2.load("./output/model.jld2")["model"]

# Loop through temperatures
for (i, temp) in enumerate(temps)
    # Add a dictionary entry for the temperature
    rhvae_dict["$(temp)"] = Dict()
    # Parse temperature value
    T = parse(Float32, replace(temp, "temp" => ""))
    # Load model
    rhvae_dict["$(temp)"]["model"] = AutoEncode.RHVAEs.RHVAE(
        deepcopy(rhvae_template.vae),
        deepcopy(rhvae_template.metric_chain),
        deepcopy(rhvae_template.centroids_data),
        deepcopy(rhvae_template.centroids_latent),
        deepcopy(rhvae_template.L),
        deepcopy(rhvae_template.M),
        T,
        deepcopy(rhvae_template.λ)
    )
    # Search for model files
    model_files = Glob.glob("./output/model_state/*$(temp)*.jld2")
    # Load parameters
    model_state = JLD2.load(model_files[end])["model_state"]
    # Set model parameters
    Flux.loadmodel!(rhvae_dict["$(temp)"]["model"], model_state)
    # Update metric
    AutoEncode.RHVAEs.update_metric!(rhvae_dict["$(temp)"]["model"])
end # for

## =============================================================================

println("Computing latent space metric...")

# Define number of points per axis
n_points = 100

# Define range of latent space
latent_range = Float32.(range(-5, 5, length=n_points))

# Define latent points to evaluate
z_mat = reduce(hcat, [[x, y] for x in latent_range, y in latent_range])

# Loop through temperatures
for temp in temps
    # Compute inverse metric tensor
    Ginv = AutoEncode.RHVAEs.G_inv(z_mat, rhvae_dict["$(temp)"]["model"])

    # Compute log determinant of metric tensor
    rhvae_dict["$(temp)"]["logdetG"] = reshape(
        -1 / 2 * AutoEncode.utils.slogdet(Ginv), n_points, n_points
    )
end # for

## =============================================================================

println("Loading data...")

# Define data directory
data_dir = "$(git_root())/data/Iwasawa_2022"

# Load file into memory
df_ic50 = CSV.read("$(data_dir)/iwasawa_ic50_tidy.csv", DF.DataFrame)

# Locate strains with missing values
missing_strains = unique(df_ic50[ismissing.(df_ic50.log2ic50), :strain])

# Remove data
df_ic50 = df_ic50[[x ∉ missing_strains for x in df_ic50.strain], :]

# Group data by strain and day
df_group = DF.groupby(df_ic50, [:strain, :day])

# Extract unique drugs to make sure the matrix is built correctly
drug = sort(unique(df_ic50.drug))

# Initialize matrix to save ic50 values
ic50_mat = Matrix{Float32}(undef, length(drug), length(df_group))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Sort data by stress
    DF.sort!(data, :drug)
    # Check that the stress are in the correct order
    if all(data.drug .== drug)
        # Add data to matrix
        ic50_mat[:, i] = Float32.(data.log2ic50)
    else
        println("group $i stress does not match")
    end # if
end # for

# Define number of environments
n_env = size(ic50_mat, 1)
# Define number of samples
n_samples = size(ic50_mat, 2)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, ic50_mat, dims=2)

# Center data to have mean zero and standard deviation one
ic50_std = StatsBase.transform(dt, ic50_mat)

## =============================================================================

println("Mapping data to latent spaces...")

# Loop through temperatures
for temp in temps
    # Encode data to latent space
    latent = rhvae_dict["$(temp)"]["model"].vae.encoder(ic50_std).µ
    # Add latent space to dictionary
    rhvae_dict["$(temp)"]["latent"] = latent
end # for

## =============================================================================

println("Transforming latent space data to tidy format...")

# Extract strains as ordered in ic50 matrix
strains_mat = [x.strain for x in keys(df_group)]
day_mat = [x.day for x in keys(df_group)]

# Initialize empty dataframe
df_latent = DF.DataFrame()

# Loop through temperatures
for temp in temps
    # Extract latent space
    latent = rhvae_dict["$(temp)"]["latent"]
    # Generate dataframe with corresponding metadata
    DF.append!(
        df_latent,
        DF.DataFrame(
            strain=strains_mat,
            day=day_mat,
            z1=latent[1, :],
            z2=latent[2, :],
            temp=temp
        )
    )
end # for

# Add corresponding metadata resistance value
df_latent = DF.leftjoin!(
    df_latent,
    unique(df_ic50[:, [:strain, :day, :parent, :env]]),
    on=[:strain, :day]
)

## =============================================================================

println("Setting output directories...")

# Define output directory
out_dir = "./output/fig/"

# Check if output directory exists
if !isdir("./output/")
    mkdir("./output/")
end # if

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

## =============================================================================

# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Define unique environments
envs = unique(df_latent.env)

# Define the number of rows and columns
rows = 3
cols = 3

# Loop through unique environments
for (k, env) in enumerate(envs)
    # Define file name
    fname = "$(out_dir)geodesics_$(env)env.pdf"

    # Group environment data by strain
    local df_group = DF.groupby(df_latent[(df_latent.env.==env), :], :strain)

    # Loop through strains
    for (j, data_strain) in enumerate(df_group)

        # Initialize plot
        fig = Figure(size=(1_000, 1_000))

        # Loop through temperatures
        for (i, temp) in enumerate(temps)
            # Extract temperature data and sort by day
            data = DF.sort!(data_strain[(data_strain.temp.==temp), :], :day)

            # Extract group metadata
            env = data.env[1]
            temp = data.temp[1]
            strain = data.strain[1]

            println("Plotting $(strain) | $(env) env | T = $(temp)")

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Plot latent space metric
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Get log determinant of metric tensor
            logdetG = rhvae_dict["$(temp)"]["logdetG"]

            # Calculate row and column indices
            row = (i - 1) ÷ cols + 1
            col = (i - 1) % cols + 1

            # Add axis to plot
            ax = Axis(
                fig[row, col],
                xlabel="latent dimension 1",
                ylabel="latent dimension 2",
            )
            # Plot heatmap of log determinant of metric tensor
            hm = heatmap!(
                ax, latent_range, latent_range, logdetG, colormap=:tokyo
                # colormap=:grays
            )

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Plot IC50 values
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Add scatter plot to axis
            scatterlines!(
                ax,
                data.z1,
                data.z2,
                markersize=8,
                linewidth=2,
                label=data.strain[1],
                color=Antibiotic.viz.colors()[:blue]
            )

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Plot geodesic
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Search for model files
            file = Glob.glob(
                "./output/geodesic_state/neuralgeodesic_$(temp)_$(env)env_" *
                "$(replace(strain, " " => "-"))strain*"
            )[1]
            # Load file
            nng_dict = JLD2.load(file)
            # Define NeuralGeodesic model
            nng = NG.NeuralGeodesic(
                nng_template.mlp,
                nng_dict["z_init"],
                nng_dict["z_end"],
            )
            # Load parameters
            Flux.loadmodel!(nng, nng_dict["model_state"])

            # Generate curve
            curve = nng(t_array)

            # Add geodesic to axis
            lines!(
                ax,
                curve[1, :],
                curve[2, :],
                linewidth=3,
                linestyle=Linestyle([0.5, 1.0, 1.5, 2.5]),
                color=:white
            )

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Add start and end points
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Add first point 
            scatter!(
                ax,
                [data.z1[1]],
                [data.z2[1]],
                color=:white,
                markersize=18,
                marker=:xcross
            )
            scatter!(
                ax,
                [data.z1[1]],
                [data.z2[1]],
                color=:black,
                markersize=10,
                marker=:xcross
            )

            # Add last point
            scatter!(
                ax,
                [data.z1[end]],
                [data.z2[end]],
                color=:white,
                markersize=18,
                marker=:utriangle
            )
            scatter!(
                ax,
                [data.z1[end]],
                [data.z2[end]],
                color=:black,
                markersize=10,
                marker=:utriangle
            )

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Add plot annotations
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Set subplot title
            ax.title = "T = $(replace(temp, "temp" => "")) | $(env) env"

        end # for temps

        # Set figure title
        Label(fig[0, :], "$(data_strain.strain[1])", fontsize=20)

        # Save figure into temporary PDF
        save("$(out_dir)temp.pdf", fig)

        # Merge PDFs
        PDFmerger.append_pdf!(
            fname, "$(out_dir)temp.pdf", cleanup=true
        )
    end # for strains
end # for envs

## =============================================================================