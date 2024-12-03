## =============================================================================
println("Loading packages...")

# Import project package
import Antibiotic

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import Glob to list files
import Glob

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes

# Import basic math libraries
import StatsBase
import LinearAlgebra
import Random

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
Antibiotic.viz.theme_makie!()

## =============================================================================

# Define output directory
fig_dir = "$(git_root())/fig/supplementary"

# Generate output directory if it doesn't exist
if !isdir(fig_dir)
    println("Generating output directory...")
    mkpath(fig_dir)
end

## =============================================================================

println("Loading raw data...")

# Load data into a DataFrame
df = CSV.read(
    "$(git_root())/data/Iwasawa_2022/iwasawa_tidy.csv", DF.DataFrame
)

# Remove blank measurements
df = df[.!df.blank, :]
# Remove zero concentrations
df = df[df.concentration_ugmL.>0, :]

## =============================================================================

# Define data to use
data = df[(df.antibiotic.=="KM").&(df.env.=="Parent_in_KM").&(df.strain_num.==13).&.!(df.blank).&(df.concentration_ugmL.>0), :]
# Remove blank measurement
# Group data by day
df_group = DF.groupby(data, :day)

# Initialize figure
fig = Figure(size=(500, 320))

# Add grid layout
gl = fig[1, 1] = GridLayout()

# Add grid layout for banner
gl_banner = gl[1, 1] = GridLayout()

# Add grid layout for plot
gl_plot = gl[2, 1] = GridLayout()

# ------------------------------------------------------------------------------
# Add banner

# Add box
box = Box(
    gl_banner[1, 1],
    color="#E6E6EF",
    strokecolor="#E6E6EF",
    alignmode=Mixed(; left=-60, right=-10) # Moves box to the left and right
)

# Add section title
Label(
    gl_banner[1, 1],
    "antibiotic resistance progression",
    fontsize=14,
    padding=(0, 0, 0, 0),
    halign=:left,
    tellwidth=false, # prevent column from contracting because of label size
    alignmode=Mixed(; left=-40) # Moves text to the left
)

# ------------------------------------------------------------------------------

# Add section title
ax = Axis(
    gl_plot[1, 1],
    xlabel="[antibiotic] (µg/mL)",
    ylabel="optical density (OD₆₂₀)",
    xscale=log10
)

# Define colors for plot
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1, length(df_group)))

# Loop through days
for (i, d) in enumerate(df_group)
    # Sort data by concentration
    DF.sort!(d, :concentration_ugmL)
    # Plot scatter line
    scatterlines!(
        ax, d.concentration_ugmL, d.OD, color=colors[i], label="$(first(d.day))"
    )
end # for

# Add legend to plot
gl_plot[1, 2] = Legend(
    fig, ax, "day", framevisible=false, nbanks=3, labelsize=10
)

save("$(fig_dir)/figSI_ic50_progression.pdf", fig)
save("$(fig_dir)/figSI_ic50_progression.png", fig)

fig