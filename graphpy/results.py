# general
import pandas as pd
import numpy as np

# spatial
import xarray as xa
import cartopy.crs as ccrs

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from graphpy import spatial, colours


def plot_spatial_confusion(
    comparison_xa: xa.DataArray,
    figsize: tuple[float, float] = [16, 12],
    fax: bool = None,
    cbar_pad: float = 0.1,
    presentation_format: bool = True,
    extent: tuple[float] = None,
) -> xa.Dataset:
    """Plot a spatial confusion matrix based on the predicted and ground truth variables in the xarray dataset.

    Parameters
    ----------
        xa_ds (xarray.Dataset): Input xarray dataset.
        ground_truth_var (str): Name of the ground truth variable in the dataset.
        predicted_var (str): Name of the predicted variable in the dataset.

    Returns
    -------    -------
        xarray.Dataset: Updated xarray dataset with the "comparison" variable added.
    """
    # calculate spatial confusion values and assign to new variable in Dataset
    if not fax:
        fig, ax = spatial.generate_geo_axis(figsize=figsize)
    else:
        fig, ax = fax[0], fax[1]

    # fetch meaning of values
    vals_dict = get_confusion_vals_dict()
    cmap_colors = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]

    im = comparison_xa.plot(
        ax=ax,
        cmap=mcolors.ListedColormap(cmap_colors),
        vmin=0,
        vmax=4,
        add_colorbar=True,
        transform=ccrs.PlateCarree(),
        alpha=1,
    )

    ax.set_aspect("equal")
    # remove old colorbar
    cb = im.colorbar
    cb.remove()
    colorbar = plt.colorbar(im, ax=[ax], location="right", pad=cbar_pad)
    num_ticks = len(cmap_colors)
    vmin, vmax = colorbar.vmin, colorbar.vmax
    colorbar.set_ticks(
        [vmin + (vmax - vmin) / num_ticks * (0.5 + i) for i in range(num_ticks)]
    )

    if presentation_format:
        label_style_dict = {"fontsize": 12, "color": "white", "rotation": 45}
        colours.customize_plot_colors(fig, ax)
        cbar_label_color = "white"
    else:
        label_style_dict = None
        cbar_label_color = "black"

    colorbar.set_ticklabels(list(vals_dict.keys()), color=cbar_label_color)

    format_spatial_plot(
        im,
        fig,
        ax,
        title="",
        # cbar_name="",
        cbar_dict={"cbar": False},
        label_style_dict=label_style_dict,
        presentation_format=presentation_format,
    )  # TODO: update this with new function

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    return fig, ax


def generate_spatial_confusion_matrix_da(
    predicted: xa.DataArray, ground_truth: xa.DataArray
) -> xa.DataArray:
    """Compute a spatial confusion matrix based on the predicted and ground truth xa.DataArray.

    Parameters
    ----------
    predicted (xa.DataArray): Predicted values.
    ground_truth (xa.DataArray): Ground truth values.

    Returns
    -------
    xa.DataArray: Spatial confusion matrix

    """
    # compare ground truth and predicted values
    true_positives = xa.where((predicted == 1) & (ground_truth == 1), 1, 0)
    true_negatives = xa.where((predicted == 0) & (ground_truth == 0), 2, 0)
    false_positives = xa.where((predicted == 1) & (ground_truth == 0), 3, 0)
    false_negatives = xa.where((predicted == 0) & (ground_truth == 1), 4, 0)

    category_variable = (
        true_positives + true_negatives + false_positives + false_negatives
    )

    return category_variable


def get_confusion_vals_dict():
    return {
        "No Data": 0,
        "True Positives": 1,
        "True Negatives": 2,
        "False Positives": 3,
        "False Negatives": 4,
    }


def plot_spatial_residuals(
    prediction_xa_da=xa.DataArray,
    ground_truth_xa_da=xa.DataArray,
    figsize: tuple[float, float] = (16, 14),
    cbar_pad: float = 0.1,
    extent: tuple[float] = None,
) -> None:
    """
    Plot the spatial differences between predicted and ground truth data.

    Parameters
    ----------
        xa_pred (xa.DataArray): Predicted data.
        xa_gt (xa.DataArray): Ground truth data.
        figsize (tuple[float, float], optional): Figure size. Default is (16, 9).

    Returns
    -------
        None
    """
    xa_diff = (ground_truth_xa_da - prediction_xa_da).rename("residuals")

    f, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})

    if xa_diff.sum() == 0:
        cbar_type = "seq"
    else:
        cbar_type = "res"

    spatial.plot_spatial(
        xa_diff,
        cbar_dict={
            "orientation": "horizontal",
            "cbar_name": "residuals",
            "cmap_type": cbar_type,
        },
        fax=(f, ax),
    )
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return f, ax


def plot_spatial_confusion_matrix(
    prediction_xa_da: xa.DataArray,
    ground_truth_xa_da: xa.DataArray,
    y: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.25,
    presentation_format: bool = True,
    extent: tuple[float] = None,
):
    # Threshold the values while retaining NaNs
    thresholded_predictions_values = np.where(
        np.isnan(prediction_xa_da),
        np.nan,
        np.where(prediction_xa_da > threshold, 1, 0),
    ).astype(int)
    thresholded_gt_values = np.where(
        np.isnan(ground_truth_xa_da),
        np.nan,
        np.where(ground_truth_xa_da > threshold, 1, 0),
    ).astype(int)

    confusion_values = generate_spatial_confusion_matrix_da(
        predicted=thresholded_predictions_values,
        ground_truth=thresholded_gt_values,
    )

    fig, ax = plot_spatial_confusion(
        confusion_values, presentation_format=presentation_format, extent=extent
    )

    return confusion_values, fig, ax


def plot_permutation_importance(
    perm_imp_df: pd.DataFrame,
    n_samples: int = 10,
    split: bool = True,
    figsize: tuple[float] = (12, 5),
):
    # compute mean and std of permutation importance
    perm_imp_stats_df = perm_imp_df.aggregate(["mean", "std"])
    ordered_df = perm_imp_stats_df.sort_values(by="mean", axis=1)

    # low_cmap = colours.ColourMapGenerator().get_cmap("lim_blue")

    # high_cmap = colours.ColourMapGenerator().get_cmap("lim_red")
    # if splitting into most negative and positive features
    if split:
        num_pts = n_samples // 2
        low_cmap = sns.light_palette(
            colours.ColourMapGenerator().lim_blue_hexes[0],
            input="hex",
            n_colors=n_samples // 2,
        )[::-1]
        high_cmap = sns.light_palette(
            colours.ColourMapGenerator().lim_red_hexes[-1],
            input="hex",
            n_colors=n_samples // 2,
        )
        # plotting
        f, (low_ax, high_ax) = plt.subplots(1, 2, sharey=True, figsize=figsize)
        sns.boxplot(ordered_df.iloc[:, :num_pts], ax=low_ax, palette=low_cmap)
        sns.boxplot(ordered_df.iloc[:, -num_pts:], ax=high_ax, palette=high_cmap)

        # set axis limits and labels
        low_ax.set_xlim(-0.5, num_pts - 0.5)
        high_ax.set_xlim(
            len(ordered_df.iloc[:, -num_pts:].columns) - 0.5 - num_pts,
            len(ordered_df.iloc[:, -num_pts:].columns),
        )
        low_ax.set_xticklabels(rotation=80, labels=ordered_df.iloc[:, :num_pts].columns)
        high_ax.set_xticklabels(
            rotation=80, labels=ordered_df.iloc[:, -num_pts:].columns
        )

        # hide the lines and ticks on low_ax and high_ax
        low_ax.spines["right"].set_visible(False)
        high_ax.spines["left"].set_visible(False)
        high_ax.tick_params(left=False)
        low_ax.tick_params(labeltop=False)  # don't put tick labels at the top
        high_ax.tick_params(labeltop=False)
        high_ax.xaxis.tick_bottom()

        d = 0.015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=low_ax.transAxes, color="k", clip_on=False)
        low_ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        low_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal

        kwargs.update(transform=high_ax.transAxes)  # switch to the bottom axes
        high_ax.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
        high_ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        for a in [low_ax, high_ax]:
            a.hlines(0, -0.5, n_samples, ls="--", color="grey")
        low_ax.set_ylabel("Permutation importance")

    else:
        low_cmap = sns.light_palette(
            spatial.colours.ColourMapGenerator().lim_blue_hexes[0],
            input="hex",
            n_colors=n_samples,
        )[::-1]
        f, ax = plt.subplots(figsize=figsize)
        sns.boxplot(ordered_df.iloc[:, :n_samples], ax=ax, palette=low_cmap)
        # formatting
        ax.set_xticklabels(rotation=80, labels=ordered_df.columns[:n_samples])
        ax.set_ylabel("Permutation importance")
        ax.hlines(0, -0.5, n_samples, ls="--", color="grey")

    return f


# DEPRECATED
# def plot_confusion_matrix(
#     labels,
#     predictions,
#     label_threshold: float = 0,
#     fax=None,
#     colorbar: bool = False,
#     presentation_format: bool = False,
# ) -> None:
#     """
#     Plot the confusion matrix.

#     Parameters
#     ----------
#         labels (Any): True labels for comparison.
#         predictions (Any): Predicted labels.
#         label_threshold (float, optional): Label threshold. Defaults to 0.
#         fax (Optional[Any], optional): Axes to plot the confusion matrix. Defaults to None.
#         colorbar (bool, optional): Whether to show the colorbar. Defaults to False.

#     Returns
#     -------
#         None
#     """
#     cmap = spatial.colours.ColourMapGenerator().get_cmap("seq")

#     if not utils.check_discrete(predictions) or not utils.check_discrete(labels):
#         labels = ml_processing.cont_to_class(labels, threshold=label_threshold)
#         predictions = ml_processing.cont_to_class(
#             predictions, threshold=label_threshold
#         )

#     classes = ["coral absent", "coral present"]
#     # initialise confusion matrix
#     cm = sklmetrics.confusion_matrix(
#         labels, predictions, labels=[0, 1], normalize="true"
#     )
#     disp = sklmetrics.ConfusionMatrixDisplay(
#         confusion_matrix=cm, display_labels=classes
#     )
#     if not fax:
#         fax = plt.subplots()

#     disp.plot(cmap=cmap, colorbar=colorbar, text_kw={"color": "black"}, ax=fax[1])

#     if presentation_format:
#         spatial.colours.customize_plot_colors(fax[0], fax[1])
