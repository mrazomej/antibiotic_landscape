using Measures, CairoMakie
import Makie
import ColorSchemes
import ColorTypes
import Colors
import DataFrames as DF

"""
    `pboc_plotlyjs!()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `plotly` backend.
"""
function pboc_plotlyjs!()
    plotlyjs(
        background_color="#E3DCD0",
        background_color_outside="white",
        foreground_color_grid="#ffffff",
        gridlinewidth=0.5,
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=8,
        dpi=300,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=(0.8, 0.8),
        foreground_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        fmt=:png
    )
end

"""
    `pboc_pyplot()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `pyplot` backend.
"""
function pboc_pyplot!()
    pyplot(
        background_color="#E3DCD0",
        background_color_outside="white",
        foreground_color_grid="#ffffff",
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=8,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=(0.8, 0.8),
        foreground_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        grid=true,
        gridcolor="white",
        gridlinewidth=1.5
    )
end

"""
    `pboc_gr()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `gr` backend.
"""
function pboc_gr!()
    gr(
        background_color="#E3DCD0",
        background_color_outside="white",
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=10,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=:topright,
        background_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        grid=true,
        gridcolor="white",
        minorgridcolor="white",
        gridlinewidth=1.5,
        minorgridlinewidth=1.5,
        bottom_margin=5mm
    )
end

"""
    `pboc_makie()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `makie` plotting library. This can be for either the GLMakie or the
CairoMakie backends
"""
function pboc_makie!()
    # if ~isfile(assetpath("fonts", "Lucida-sans-Unicode-Regular.ttf"))
    #     @warn "Lucida sans Unicode Regular font not added to Makie Fonts. Add to `~/.julia/packages/Makie/gQOQF/assets/fonts/`. Defaulting to NotoSans."
    #     Font = assetpath("fonts", "NotoSans-Regular.tff")
    # else
    #     Font = assetpath("fonts", "Lucida-Sans-Unicode-Regular.ttf")
    # end

    Font = "Lucida Sans Regular"
    # Seaborn colorblind
    colors = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9"
    ]

    theme = Theme(
        Axis=(
            backgroundcolor="#E3DCD0",

            # Font sizes
            titlesize=16,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=14,
            yticklabelsize=14,

            # Font styles
            titlefont=Font,
            xticklabelfont=Font,
            yticklabelfont=Font,
            xlabelfont=Font,
            ylabelfont=Font,

            # Grid
            xgridwidth=1.25,
            ygridwidth=1.25,
            xgridcolor="white",
            ygridcolor="white",
            xminorgridcolor="white",
            yminorgridcolor="white",
            xminorgridvisible=false,
            xminorgridwidth=1.0,
            yminorgridvisible=false,
            yminorgridwidth=1.0,

            # Axis ticks
            minorticks=false,
            xticksvisible=false,
            yticksvisible=false,

            # Box
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            bottomspinevisible=false,
        ),
        Legend=(
            titlesize=15,
            labelsize=15,
            bgcolor="#E3DCD0",
        ),
        backgroundcolor="white",
        linewidth=1.25,
    )
    set_theme!(theme)
end

@doc raw"""
    `theme_makie()`

Set plotting default to personal style for the `makie` plotting library. This
can be for either the GLMakie or the CairoMakie backends.
"""
function theme_makie!()
    # Seaborn colorblind
    colors = ColorSchemes.seaborn_colorblind

    theme = Theme(
        fonts=(;
            regular="Roboto Light",
            bold="Roboto Regular",
            italic="Roboto Light Italic",
            bold_italic="Roboto Regular Italic",
            extra_bold="Roboto Bold",
            extra_bold_italic="Roboto Bold Italic"
        ),
        Figure=(
            resolution = (300, 300)
        ),
        Axis=(
            # backgroundcolor="#EAEAF2", 
            backgroundcolor="#E6E6EF",

            # Font sizes
            titlesize=16,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=14,
            yticklabelsize=14,

            # Font styles
            titlefont=:bold,
            xticklabelfont=:regular,
            yticklabelfont=:regular,
            xlabelfont=:regular,
            ylabelfont=:regular,

            # Grid
            xgridwidth=1.25,
            ygridwidth=1.25,
            xgridcolor="white",
            ygridcolor="white",
            xminorgridcolor="white",
            yminorgridcolor="white",
            xminorgridvisible=false,
            xminorgridwidth=1.0,
            yminorgridvisible=false,
            yminorgridwidth=1.0,

            # Axis ticks
            minorticks=false,
            xticksvisible=false,
            yticksvisible=false,

            # Box
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            bottomspinevisible=false,
        ),
        Legend=(
            titlesize=15,
            labelsize=15,
            bgcolor="#E6E6EF",
        ),
        Lines=(
            linewidth=2,
        ),
        backgroundcolor="white",
        linewidth=1.25,
    )
    set_theme!(theme)
end

@doc raw"""
    colors()

Returns dictionary with personal color palette.
"""
function colors()
    col = Dict(
        :dark_black => "#000000",
        :black => "#000000",
        :light_black => "#05080F",
        :pale_black => "#1F1F1F",
        :dark_blue => "#2957A8",
        :blue => "#3876C0",
        :light_blue => "#81A9DA",
        :pale_blue => "#C0D4ED",
        :dark_green => "#2E5C0A",
        :green => "#468C12",
        :light_green => "#6EBC24",
        :pale_green => "#A9EB70",
        :dark_red => "#912E27",
        :red => "#CB4338",
        :light_red => "#D57A72",
        :pale_red => "#E8B5B0",
        :dark_gold => "#B68816",
        :gold => "#EBC21F",
        :light_gold => "#F2D769",
        :pale_gold => "#F7E6A1",
        :dark_purple => "#5E315E",
        :purple => "#934D93",
        :light_purple => "#BC7FBC",
        :pale_purple => "#D5AFD5"
    )

    # Initialize dictionary
    colors = Dict()

    # Loop through elements
    for (key, item) in col
        # Convert element to dictionary
        setindex!(colors, Colors.color(item), key)
    end # for

    return colors
end # function

@doc raw"""
    `corner_plot!(
        fig, df, plot_var, plot_value, group_var, color;
        colgap=2, rowgap=2
    )`

Function to generate a corner plot for different variables using `Makie.jl`.

# Arguments
- `fig::Makie.Figure`: Figure where the plot will be inserted.
- `df::DataFrames.DataFrame`: **tidy** dataframe with data to be plotted.
- `plot_var::Symbol`: Name of column indicating the variable names that will be
  plot.
- `plot_value::Symbol`: Name of the column with the numerical value to be plot.
- `group_var::Union{Vector{Symbol}, Symbol}`: Name of column by which to group
  data when plotting.
- `color`: Single color or list of colors to be used for plot.

## Optional arguments
- `marker::Symbol`: Marker to use in scatter plots
- `markersize::Real`: Marker size for scatter plot.
- `colgap::Real=2`: gap between subplot columns.
- `rowgap::Real=2`: gap between subplot rows.
- `density_color=ColorSchemes.seaborn_colorblind[1]`: Color to be used for the
  density plots.
- `legend::Bool=true`: Boolean indicating if legend should be added to the plot
  for each of the groups defined by `group_var`.
"""
function corner_plot!(
    fig::Makie.Figure,
    df::DF.DataFrame,
    plot_var::Symbol,
    plot_value::Symbol,
    group_var::Union{Vector{Symbol},Symbol},
    colors;
    marker::Symbol=:circle,
    markersize::Real=8,
    colgap::Real=2,
    rowgap::Real=2,
    density_color=ColorSchemes.seaborn_colorblind[1],
    legend::Bool=true
)
    # Define variable names
    var_names = unique(df[!, plot_var])
    # Define number of variables
    n_var = length(var_names)

    # Initialize grid layout where axes will exist
    gl = GridLayout(fig[1, 1])
    # Initialize object to save axes
    axes = Dict()
    # Loop through variables
    for i = 1:n_var
        # Loop through axes
        for j = 1:i
            # Add axes
            axes[[i, j]] = Axis(gl[i, j])
            # Remove decorations
            hidedecorations!(axes[[i, j]], label=false)
        end # for
    end # for
    # Change gap between plots
    colgap!(gl, colgap)
    rowgap!(gl, rowgap)

    # Collect plot keys
    ax_keys = sort(collect(keys(axes)))

    # Group data
    df_group = DF.groupby(df, group_var)
    # Collect group keys
    df_keys = [first(x) for x in collect(keys(df_group))]

    # Loop through keys
    for k in ax_keys
        # Extract values
        i, j = k

        # Check which plot should be done
        if i == j
            # Plot density
            density!(
                axes[k],
                df[df[:, plot_var].==var_names[i], plot_value],
                color=density_color,
            )
        else
            # Loop through groups
            for (idx, data) in enumerate(df_group)
                # Extract x axis
                x_val = data[data[:, plot_var].==var_names[i], plot_value]
                # Extract y axis
                y_val = data[data[:, plot_var].==var_names[j], plot_value]
                # Plot scatter plot
                scatter!(
                    axes[k],
                    x_val,
                    y_val,
                    color=colors[idx],
                    marker=marker,
                    markersize=markersize
                )
            end # for
        end # if

        # Add y-axis labels
        if (j == 1)
            axes[k].ylabel = var_names[i]
        end

        # Add x-axis labels
        if (i == n_var)
            axes[k].xlabel = var_names[j]
        end # if
    end # for

    # Check if legend should be added
    if legend
        # Define elements to be added to legend
        leg_sym = [
            MarkerElement(
                marker=marker,
                color=colors[i],
                strokecolor=:transparent,
                markersize=markersize
            ) for i = 1:length(df_group)
        ]
        # Add legend
        Legend(gl[1, end], leg_sym, df_keys, halign=:left, valign=:top)
    end # if
end # function