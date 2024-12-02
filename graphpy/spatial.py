# general
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# spatial
import xarray as xa
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom
from graphpy import colours


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10), map_proj=ccrs.PlateCarree(), dpi=300
):
    return plt.figure(figsize=figsize, dpi=dpi), plt.axes(projection=map_proj)


def plot_spatial(
    xa_da: xa.DataArray,
    fax: Axes = None,
    title: str = "default",
    figsize: tuple[float, float] = (10, 10),
    val_lims: tuple[float, float] = None,
    presentation_format: bool = False,
    dpi: int = 300,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
    map_proj=ccrs.PlateCarree(),
    alpha: float = 1,
    extent: list[float] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a spatial plot with colorbar, coastlines, landmasses, and gridlines.

    Parameters
    ----------
    xa_da (xa.DataArray): The input xarray DataArray representing the spatial data.
    title (str, optional): The title of the plot.
    cbar_name (str, optional): The name of the DataArray.
    val_lims (tuple[float, float], optional): The limits of the colorbar range.
    cmap_type (str, optional): The type of colormap to use.
    symmetric (bool, optional): Whether to make the colorbar symmetric around zero.
    edgecolor (str, optional): The edge color of the landmasses.
    orientation (str, optional): The orientation of the colorbar ('vertical' or 'horizontal').
    labels (list[str], optional): Which gridlines to include, as strings e.g. ["t","r","b","l"]
    map_proj (str, optional): The projection of the map.
    extent (list[float], optional): The extent of the plot as [min_lon, max_lon, min_lat, max_lat].

    Returns
    -------
    tuple: The figure and axes objects.
    TODO: saving option and tidy up presentation formatting
    """
    # may need to change this
    # for some reason fig not including axis ticks. Universal for other plotting
    if not fax:
        if extent == "global":
            fig, ax = generate_geo_axis(
                figsize=figsize, map_proj=ccrs.Robinson(), dpi=dpi
            )
            ax.set_global()
        else:
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=map_proj, dpi=dpi)

    else:
        fig, ax = fax[0], fax[1]

    if isinstance(extent, list):
        ax.set_extent(extent, crs=map_proj)

    default_cbar_dict = {
        "cbar_name": None,
        "cbar": True,
        "orientation": "vertical",
        "cbar_pad": 0.1,
        "cbar_frac": 0.025,
        "cmap_type": "seq",
    }

    if cbar_dict:
        for k, v in cbar_dict.items():
            default_cbar_dict[k] = v
        if val_lims:
            default_cbar_dict["extend"] = "both"

    # if not cbarn_name specified, make name of variable
    cbar_name = default_cbar_dict["cbar_name"]
    if isinstance(xa_da, xa.DataArray) and not cbar_name:
        cbar_name = xa_da.name

    # if title not specified, make title of variable at resolution
    # if title: # TODO: do I want this specified?
    #     if title == "default":
    #         resolution_d = np.mean(spatial_data.calculate_spatial_resolution(xa_da))
    #         resolution_m = np.mean(spatial_data.degrees_to_distances(resolution_d))
    #         title = (
    #             f"{cbar_name} at {resolution_d:.4f}° (~{resolution_m:.0f} m) resolution"
    #         )

    # if colorbar limits not specified, set to be maximum of array
    if not val_lims:  # TODO: allow dynamic specification of only one of min/max
        vmin, vmax = np.nanmin(xa_da.values), np.nanmax(xa_da.values)
    else:
        vmin, vmax = min(val_lims), max(val_lims)

    if (
        default_cbar_dict["cmap_type"] == "div"
        or default_cbar_dict["cmap_type"] == "res"
    ):
        if vmax < 0:
            vmax = 0.01
        cmap, norm = colours.ColourMapGenerator().get_cmap(
            default_cbar_dict["cmap_type"], vmin, vmax
        )
    else:
        cmap = colours.ColourMapGenerator().get_cmap(default_cbar_dict["cmap_type"])

    im = xa_da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,  # for further formatting later
        transform=ccrs.PlateCarree(),
        alpha=alpha,
        norm=(
            norm
            if (
                default_cbar_dict["cmap_type"] == "div"
                or default_cbar_dict["cmap_type"] == "res"
            )
            else None
        ),
    )

    if presentation_format:
        fig, ax = colours.customize_plot_colors(fig, ax)
        # ax.tick_params(axis="both", which="both", length=0)

    # nicely format spatial plot
    format_spatial_plot(
        image=im,
        fig=fig,
        ax=ax,
        title=title,
        # cbar_name=cbar_name,
        # cbar=default_cbar_dict["cbar"],
        # orientation=default_cbar_dict["orientation"],
        # cbar_pad=default_cbar_dict["cbar_pad"],
        # cbar_frac=default_cbar_dict["cbar_frac"],
        cartopy_dict=cartopy_dict,
        presentation_format=presentation_format,
        labels=labels,
        cbar_dict=default_cbar_dict,
        label_style_dict=label_style_dict,
    )

    return fig, ax, im


def format_cbar(image, fig, ax, cbar_dict, labels: list[str] = ["l", "b"]):

    if cbar_dict["orientation"] == "vertical":
        cbar_rect = [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    else:
        cbar_rect = [
            ax.get_position().x0,
            ax.get_position().y0 - 0.04,
            ax.get_position().width,
            0.02,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    cax = fig.add_axes(cbar_rect)

    cb = plt.colorbar(
        image,
        orientation=cbar_dict["orientation"],
        label=cbar_dict["cbar_name"],
        cax=cax,
        extend=cbar_dict["extend"] if "extend" in cbar_dict else "neither",
    )
    if cbar_dict["orientation"] == "horizontal":
        cbar_ticks = cb.ax.get_xticklabels()
    else:
        cbar_ticks = cb.ax.get_yticklabels()

    return cb, cbar_ticks, labels


def format_cartopy_display(ax, cartopy_dict: dict = None):

    default_cartopy_dict = {
        "category": "physical",
        "name": "land",
        "scale": "10m",
        "edgecolor": "black",
        "facecolor": "#cfcfcf",  # "none"
        "linewidth": 0.5,
        "alpha": 0.5,
    }

    if cartopy_dict:
        for k, v in cartopy_dict.items():
            default_cartopy_dict[k] = v

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            default_cartopy_dict["category"],
            default_cartopy_dict["name"],
            default_cartopy_dict["scale"],
            edgecolor=default_cartopy_dict["edgecolor"],
            facecolor=default_cartopy_dict["facecolor"],
            linewidth=default_cartopy_dict["linewidth"],
            alpha=default_cartopy_dict["alpha"],
        ),
        zorder=100,
    )

    return ax


def format_spatial_plot(
    image: xa.DataArray,
    fig: Figure,
    ax: Axes,
    title: str = None,
    # cbar: bool = True,
    # cmap_type: str = "seq",
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
) -> tuple[Figure, Axes]:
    """Format a spatial plot with a colorbar, title, coastlines and landmasses, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.
        cbar_name (str): label of colorbar.
        cbar (bool): whether to include a colorbar.
        orientation (str): orientation of colorbar.
        cbar_pad (float): padding of colorbar.
        edgecolor (str): color of landmass edges.
        presentation_format (bool): whether to format for presentation.
        labels (list[str]): which gridlines to include, as strings e.g. ["t","r","b","l"]
        label_style_dict (dict): dictionary of label styles.

    Returns
    -------
        Figure, Axes
    """
    if cbar_dict and cbar_dict["cbar"]:
        cb, cbar_ticks, labels = format_cbar(image, fig, ax, cbar_dict, labels)

    ax = format_cartopy_display(ax, cartopy_dict)
    ax.set_title(title)

    # format ticks, gridlines, and colours
    ax.tick_params(axis="both", which="major")
    default_label_style_dict = {
        "fontsize": 12,
        "color": "black",
        "rotation": 45,
        # "gridlines": True,
    }

    # if default_label_style_dict:
    if label_style_dict:
        for k, v in label_style_dict.items():
            default_label_style_dict[k] = v

    # if default_label_style_dict["gridlines"]:
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        # x_inline=False, y_inline=False
    )

    gl.xlabel_style = default_label_style_dict
    gl.ylabel_style = default_label_style_dict

    if (
        not labels
    ):  # if no labels specified, set up something to iterate through returning nothing
        labels = [" "]
    if labels:
        # convert labels to relevant boolean: ["t","r","b","l"]
        gl.top_labels = "t" in labels
        gl.bottom_labels = "b" in labels
        gl.left_labels = "l" in labels
        gl.right_labels = "r" in labels
    if presentation_format:
        default_label_style_dict["color"] = "white"
        if cbar_dict and cbar_dict["cbar"]:
            plt.setp(cbar_ticks, color="white")
            cb.set_label(cbar_dict["cbar_name"], color="white")

    return fig, ax


# def plot_var_mask(
#     xa_d: xa.Dataset | xa.DataArray,
#     limits: tuple[float] = [-2000, 0],
#     title: str = "masked variable",
# ) -> None:
#     # plot shallow water mask
#     shallow_mask = spatial_data.generate_var_mask(xa_d)
#     plot_spatial(shallow_mask, cmap_type="lim_blue", title=title, cbar=False)
#     return shallow_mask


def plot_train_test_datasets(
    train_da: xa.DataArray,
    test_da: xa.DataArray,
    text_pos: tuple[float, float] = (0.5, 0.5),
    figsize: tuple[float, float] = (10, 10),
    binarise: bool = False,
    dpi: int = 300,
    fax: tuple = None,
    **graph_kwargs: dict,
):
    if binarise:  # if just showing which is which (not values)
        # cast first to ones, second to 0s
        train_ds = train_da.where(~train_da.notnull(), 1)
        test_ds = test_da.where(~test_da.notnull(), 0)

    if not fax:
        f, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=figsize, dpi=dpi
        )
    else:
        f, ax = fax
    plot_spatial(
        train_ds,
        title="",
        fax=(f, ax),
        val_lims=[0, 1],
        cbar_dict={"cbar": False},
        **graph_kwargs,
    )
    plot_spatial(
        test_ds,
        title="",
        fax=(f, ax),
        val_lims=[0, 1],
        cbar_dict={"cbar": False},
        **graph_kwargs,
    )

    # # rough-and-ready, not particularly elegant (also only works for case study area):
    # # TODO: generalise
    # train_rect = patches.Rectangle((141, -25), 2, 1, facecolor='#d83c04', zorder=100)
    # ax.add_patch(train_rect)
    # test_rect = patches.Rectangle((141, -27), 2, 1, facecolor='#3B9AB2', zorder=100)
    # ax.add_patch(test_rect)
    # ax.text(144, -25, 'train dataset', verticalalignment='bottom', color='#d83c04', zorder=100)
    # ax.text(144, -27, 'test dataset', verticalalignment='bottom', color='#3B9AB2', zorder=100)
    # # plt.show()
