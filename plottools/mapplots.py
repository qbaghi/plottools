import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


def rectangular_plot(x_arr, y_arr, z, xlabel='', ylabel='', log_norm=True):

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(y_arr, x_arr)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    # z = z[:-1, :-1]
    #

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    # cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap()
    if not log_norm:
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    else:
        norm = LogNorm(vmin=z.min(), vmax=z.max())

    fig, ax = plt.subplots(nrows=1)
    # im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    im = ax.pcolormesh(x, y, z, norm=norm)
    fig.colorbar(im, ax=ax)
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig, ax