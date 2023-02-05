import numpy as np
import matplotlib.pyplot as plt

def add_yline(ax, yval, xrange, **kwargs):
    """Add a horrionzal line to the plot."""
    nvals = 100
    ax.plot(np.linspace(xrange[0], xrange[1], nvals), 
            [yval]*nvals, **kwargs)
    return ax