# plotting
import matplotlib.pyplot as plt


def grid_subplots(total, wrap=None, **kwargs):
    if wrap is not None:
        cols = min(total, wrap)
        rows = 1 + (total - 1) // wrap
    else:
        cols = total
        rows = 1
    fig, ax = plt.subplots(rows, cols, **kwargs)
    return fig, ax
