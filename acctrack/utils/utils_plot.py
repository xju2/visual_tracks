from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def add_yline(ax: plt.Axes, yval: float, xrange: Tuple[float, float],
              **kwargs) -> plt.Axes:
    """Add a horrionzal line to the plot."""
    nvals = 100
    ax.plot(np.linspace(xrange[0], xrange[1], nvals),
            [yval] * nvals, **kwargs)
    return ax

def add_mean_std(array: np.ndarray,
                 x: float, y: float,
                 ax: plt.Axes,
                 color='k', dy=0.3, digits=2, fontsize=12, with_std=True) -> plt.Axes:
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(x, y, "mean: {0:.{1}f}".format(this_mean, digits),
            color=color, fontsize=fontsize)
    if with_std:
        ax.text(x, y - dy, "std: {0:.{1}f}".format(this_std, digits),
                color=color, fontsize=fontsize)
        return ax
    return ax
