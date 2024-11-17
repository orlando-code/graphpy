# general
from tqdm import tqdm

# spatial
import xarray as xa
import cartopy.crs as ccrs

# file handling
from pathlib import Path

# plotting
import matplotlib.pyplot as plt
from matplotlib import animation

# custom
from graphpy import spatial


def duration_to_interval(num_frames: int, duration: int = 5000) -> int:
    """Given the number of frames and duration of a gif in milliseconds, calculates the frame interval.

    Parameters
    ----------
    num_frames (int): The number of frames in the gif.
    duration (int, optional): The duration of the gif in milliseconds. Default is 5000ms (5 seconds).

    Returns
    -------
    int: The frame interval for the gif in milliseconds.
    """
    return duration / num_frames


def generate_like_variable_timeseries_gifs(
    xa_ds: xa.Dataset,
    variables: list[str] = None,
    start_end_freq: tuple[str, str, str] = None,
    variable_dict: dict = None,
    interval: int = 500,
    duration: int = None,
    repeat_delay: int = 5000,
    dest_dir_path: Path | str = None,
) -> dict:
    """Wrapper for generate_variable_timeseries_gif allowing generation of a set of similar gifs for each specified
    variable in an xarray Dataset

    Parameters
    ----------
    xa_das (xr.Dataset): The xarray Dataset array containing variables to animate.
    variables (list[str], optional): Choice of variables to animate. Defaults to None (all variablees animated).
    start_end_freq (tuple of str, optional) A tuple containing the start date, end date, and frequency to index the
        "time" coordinate. If not provided, the whole time coordinate will be used.
    variable_dict (dict, optional): A dictionary with keys as the original variable names and values as the names to be
        displayed in the plot. If not provided, the original variable name will be used.
    interval (int, optional): The delay between frames in milliseconds.
    duration (int, optional): The duration of the GIF in milliseconds. If provided, the frame interval will be
        calculated automatically based on the number of frames and the duration.
    repeat_delay (int, optional): The delay before the animation repeats in milliseconds.
    dest_dir_path (Union[pathlib.Path, str], optional): The directory to save the output GIF. If not provided, the
        current working directory will be used.

    Returns
    -------
    list[animation.FuncAnimation]: list containing the animation objects.
    """
    # if no specific variables specified, animate all available
    if not variables:
        variables = list(xa_ds.data_vars)

    ani_dict = {}
    for var in tqdm(variables):
        ani = generate_variable_timeseries_gif(
            xa_ds[var],
            start_end_freq=start_end_freq,
            variable_dict=variable_dict,
            interval=interval,
            duration=duration,
            repeat_delay=repeat_delay,
            dest_dir_path=dest_dir_path,
        )
        ani_dict[var] = ani

    return ani_dict


# TODO: fix this, if useful
def generate_variable_timeseries_gif(
    xa_da: xa.DataArray,
    start_end_freq: tuple[str, str, str] = None,
    variable_dict: dict = None,
    interval: int = 500,
    duration: int = None,
    repeat_delay: int = 5000,
    dest_dir_path: Path | str = None,
) -> animation.FuncAnimation:
    # TODO: not yet tested
    """Create an animation showing the values of a variable in an xarray DataArray over time.

    Parameters
    ----------
    xa_da (xr.DataArray): The xarray DataArray containing the variable to animate.
    start_end_freq (tuple of str, optional) A tuple containing the start date, end date, and frequency to index the
        "time" coordinate. If not provided, the whole time coordinate will be used.
    interval (int, optional): The delay between frames in milliseconds.
    duration (int, optional): The duration of the GIF in milliseconds. If provided, the frame interval will be
        calculated automatically based on the number of frames and the duration.
    repeat_delay (int, optional): The delay before the animation repeats in milliseconds.
    dest_dir_path (Union[pathlib.Path, str], optional): The directory to save the output GIF. If not provided, the
        current working directory will be used.

    Returns
    -------
    animation.FuncAnimation: The animation object.

    TODO: Fix the time resampling function
    """
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax, _, _, _ = spatial.plot_spatial(xa_da, fax=(fig, ax))
    fig.tight_layout()

    variable_name = xa_da.name

    def update(i, variable_name=variable_name) -> None:
        """Updates a Matplotlib plot with a new frame.

        Parameters
        ----------
        i (int): The index of the frame to display.
        variable_name (str): The name of the variable being plotted.

        Returns
        -------
        None
        TODO: add single colorbar (could delete previous one each time, somehow)
        """
        cb = True
        timestamp = spatial_data.date_from_dt(xa_da.time[i].values)
        ax.set_title(f"{variable_name}\n{timestamp}")
        if i > 1:
            cb = False
        xa_da.isel(time=i).plot(ax=ax, add_colorbar=cb, cmap="viridis")

    if start_end_freq:
        # temporally resample DataArray
        xa_da, freq = spatial_data.resample_dataarray(xa_da, start_end_freq)

    # generate gif_name
    (start, end) = (
        spatial_data.date_from_dt(xa_da.time.min().values),
        spatial_data.date_from_dt(xa_da.time.max().values),
    )
    gif_name = f"{variable_name}_{start}_{end}"

    # if duration_specified
    if duration:
        interval = duration_to_interval(num_frames=xa_da.time.size, duration=duration)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=xa_da.time.size,
        interval=interval,
        repeat_delay=repeat_delay,
        repeat=True,
        blit=False,
    )
    # TODO: put in different function
    # save animation as a gif
    writergif = animation.PillowWriter(fps=30)
    # if destination directory not provided, save in current working directory
    if not dest_dir_path:
        dest_dir_path = file_ops.guarantee_existence(
            Path().absolute() / "gif_directory"
        )
    save_path = (dest_dir_path / gif_name).with_suffix(".gif")
    ani.save(str(save_path), writer=writergif)

    print(
        f"""Gif for {variable_name} between {start} and {end} written to {save_path}.
        \nInterval: {interval}, repeat_delay: {repeat_delay}."""
    )

    return ani
