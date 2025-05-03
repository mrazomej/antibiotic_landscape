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
import DimensionalData as DD

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
import PDFmerger: append_pdf!

# Activate backend
CairoMakie.activate!()

# Set plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Locate current directory
path_dir = pwd()

# Locate current directory
path_dir = pwd()

# Find the path perfix where input data is stored
out_prefix = replace(
    match(r"processing/(.*)", path_dir).match,
    "processing" => "",
)

# Define simulation directory
sim_dir = "$(git_root())/output$(out_prefix)/sim_evo"
# Define model directory
vae_dir = "$(git_root())/output$(out_prefix)/vae"
# Define output directory
state_dir = "$(vae_dir)/model_state"
# Define directory to store trained geodesic curves
geodesic_dir = "$(vae_dir)/geodesic_state/"
# Define figure directory
fig_dir = "$(git_root())/fig$(out_prefix)/vae"

# Generate figure directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating figure directory...")
    mkpath(fig_dir)
end

## =============================================================================

# List all files in the directory
geodesic_files = Glob.glob("$(geodesic_dir)/*rhvaeepoch0075*.jld2"[2:end], "/")

# Initialize dataframe to store files metadata
df_meta = DF.DataFrame()

# Loop over geodesic state files
for gf in geodesic_files
    # Extract initial generation number from file name using regular expression
    t_init = parse(Int, match(r"timeinit(\d+)", gf).captures[1])
    # Extract final generation number from file name using regular expression
    t_final = parse(Int, match(r"timefinal(\d+)", gf).captures[1])
    # Extract lineage number from file name using regular expression
    lin = parse(Int, match(r"lineage(\d+)", gf).captures[1])
    # Extract replicate number from file name using regular expression
    rep = parse(Int, match(r"replicate(\d+)", gf).captures[1])
    # Extract evolution condition from file name using regular expression
    evo = parse(Int, match(r"evo(\d+)", gf).captures[1])
    # Extract RHVAE epoch number from file name using regular expression
    rhvae_epoch = parse(Int, match(r"rhvaeepoch(\d+)", gf).captures[1])
    # Extract geodesic epoch number from file name using regular expression
    geo_epoch = parse(Int, match(r"geoepoch(\d+)", gf).captures[1])
    # Append as DataFrame
    DF.append!(
        df_meta,
        DF.DataFrame(
            :t_init => t_init,
            :t_final => t_final,
            :lineage => lin,
            :rep => rep,
            :evo => evo,
            :rhvae_epoch => rhvae_epoch,
            :geodesic_epoch => geo_epoch,
            :geodesic_state => gf,
        ),
    )
end # for gf in geodesic_files

## =============================================================================

println("Loading NeuralGeodesic template...")
nng_template = JLD2.load("$(vae_dir)/geodesic.jld2")["model"].mlp

# Define number of points per axis
n_time = 75
# Define time points along curve
t_array = Float32.(collect(range(0, 1, length=n_time)))

## =============================================================================

# Load RHVAE model
rhvae = JLD2.load("$(vae_dir)/model.jld2")["model"]
# List parameters for epochs
param_files = sort(Glob.glob("$(state_dir)/*.jld2"[2:end], "/"))
# Load last epoch
Flux.loadmodel!(rhvae, JLD2.load(param_files[end])["model_state"])
# Update metric
AET.RHVAEs.update_metric!(rhvae)

## =============================================================================

println("Loading data into memory...")

# Define subsample interval for data
n_sub = 10

# Load fitnotype profiles
fitnotype_profiles = JLD2.load("$(sim_dir)/sim_evo.jld2")["fitnotype_profiles"]

# Extract initial and final time points
t_init, t_final = collect(DD.dims(fitnotype_profiles, :time)[[1, end]])
# Subsample time series
fitnotype_profiles = fitnotype_profiles[time=DD.At(t_init:n_sub:t_final)]

# Define number of environments
n_env = length(DD.dims(fitnotype_profiles, :landscape))

# Extract fitness data bringing the fitness dimension to the first dimension
fit_data = permutedims(fitnotype_profiles.fitness.data, (5, 1, 2, 3, 4, 6))
# Reshape the array to a Matrix
fit_data = reshape(fit_data, size(fit_data, 1), :)

# Reshape the array to stack the 3rd dimension
fit_mat = log.(fit_data)

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment 
dt = StatsBase.fit(StatsBase.ZScoreTransform, fit_mat, dims=2)

# Standardize the data to have mean 0 and standard deviation 1
log_fitnotype_std = DD.DimArray(
    mapslices(slice -> StatsBase.transform(dt, slice),
        log.(fitnotype_profiles.fitness.data),
        dims=[5]),
    fitnotype_profiles.fitness.dims,
)

## =============================================================================

println("Map data to latent space...")

# Define latent space dimensions
latent = DD.Dim{:latent}([:latent1, :latent2])

# Map data to latent space
dd_latent = DD.DimArray(
    dropdims(
        mapslices(slice -> rhvae.vae.encoder(slice).μ,
            log_fitnotype_std.data,
            dims=[5]);
        dims=1
    ),
    (log_fitnotype_std.dims[2:4]..., latent, log_fitnotype_std.dims[6]),
)

## =============================================================================

println("Compute Riemannian metric for latent space...")

# Define number of points per axis
n_points = 100

# Extract latent space ranges
latent1_range = range(
    minimum(dd_latent[latent=DD.At(:latent1)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent1)]) + 1.5,
    length=n_points
)
latent2_range = range(
    minimum(dd_latent[latent=DD.At(:latent2)]) - 1.5,
    maximum(dd_latent[latent=DD.At(:latent2)]) + 1.5,
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

# Define number of columns and rows per page
cols = 5
rows = 5
panels_per_page = cols * rows

# Group data by :evo
dd_group = DD.groupby(dd_latent, DD.dims(dd_latent, :evo))

# Calculate number of pages needed
n_groups = length(dd_group)
n_pages = ceil(Int, n_groups / panels_per_page)

# Define file name
fname = "$(fig_dir)/geodesic_latent_trajectory"

# Remove previous PDF file if it exists
if isfile("$(fname).pdf")
    rm("$(fname).pdf")
end

# Loop through pages
for page in 1:n_pages
    println("Generating page $(page) of $(n_pages)")

    # Calculate start and end indices for this page
    start_idx = (page - 1) * panels_per_page + 1
    end_idx = min(page * panels_per_page, n_groups)

    # Initialize figure for this page
    fig = Figure(size=(200 * cols, 200 * rows))
    # Add grid layout
    gl = fig[1, 1] = GridLayout()

    # Loop through evolution conditions for this page
    for (i, dd_evo) in enumerate(dd_group[start_idx:end_idx])
        # Extract evolution condition
        evo = DD.dims(dd_evo, :evo)[1]

        println("   - Plotting geodesic: $(start_idx + i - 1)")
        # Define row and column index
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        # Add axis
        ax = Axis(
            gl[row, col],
            aspect=AxisAspect(1),
            title="evolution condition $(evo)",
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

        # Group dd_evo by lineage
        dd_lineage = DD.groupby(dd_evo, DD.dims(dd_evo, :lineage))

        # Loop through lineages
        for (j, dd_lin) in enumerate(dd_lineage)
            # Extract lineage
            lin = DD.dims(dd_lin, :lineage)[1]
            # Plot lineage
            scatterlines!(
                ax,
                vec(dd_lin[latent=DD.At(:latent1), replicate=1].data),
                vec(dd_lin[latent=DD.At(:latent2), replicate=1].data),
                markersize=6,
                linewidth=2,
                color=ColorSchemes.Paired_12[j*2],
            )
        end

        # Loop through lineages
        for (j, dd_lin) in enumerate(dd_lineage)
            # Extract lineage
            lin = DD.dims(dd_lin, :lineage)[1]
            # Load geodesic state
            geo_state = JLD2.load(
                first(
                    df_meta[
                        (df_meta.evo.==evo).&(df_meta.lineage.==lin).&(df_meta.rep.==1),
                        :geodesic_state]
                )
            )
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
                linestyle=(:dot, :dense),
                color=ColorSchemes.Paired_12[j*2-1],
            )
        end

        # Loop through lineages
        for (j, dd_lin) in enumerate(dd_lineage)
            # Add first point 
            scatter!(
                ax,
                first(vec(dd_lin[latent=DD.At(:latent1), replicate=1].data)),
                first(vec(dd_lin[latent=DD.At(:latent2), replicate=1].data)),
                color=:white,
                markersize=11,
                marker=:xcross
            )
            scatter!(
                ax,
                first(vec(dd_lin[latent=DD.At(:latent1), replicate=1].data)),
                first(vec(dd_lin[latent=DD.At(:latent2), replicate=1].data)),
                markersize=7,
                marker=:xcross,
                color=ColorSchemes.Paired_12[j*2],
            )

            # Add last point
            scatter!(
                ax,
                last(vec(dd_lin[latent=DD.At(:latent1), replicate=1].data)),
                last(vec(dd_lin[latent=DD.At(:latent2), replicate=1].data)),
                color=:white,
                markersize=11,
                marker=:utriangle
            )
            scatter!(
                ax,
                last(vec(dd_lin[latent=DD.At(:latent1), replicate=1].data)),
                last(vec(dd_lin[latent=DD.At(:latent2), replicate=1].data)),
                color=ColorSchemes.Paired_12[j*2],
                markersize=7,
                marker=:utriangle
            )
        end
    end

    # Save this page as PNG and PDF
    png_file = "$(fname)_$(lpad(page, 2, '0')).png"
    # Save as PNG
    save(png_file, fig)
    # Save temporary PDF
    save("temp.pdf", fig)
    # Append to PDF
    append_pdf!("$(fname).pdf", "temp.pdf", cleanup=true)
end


## =============================================================================

println("Done!")
