"""Visualize tracks."""

import numpy as np
import matplotlib.pyplot as plt


def plot(track: np.ndarray,
        figsize=(15,12),
        fontsize=16,
        minorsize=14,
        unit='mm',
        outname=None, **kwargs) -> None:
    """Plot a track from different views"""

    if track.shape[1] != 3:
        raise ValueError("Track must have 3 columns for x, y, and z.")

    if 'lw' not in kwargs:
        kwargs['lw'] = 2

    x = track[:, 0]
    y = track[:, 1]
    z = track[:, 2]
    r = np.sqrt(x**2 + y**2)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='polar')


    ## plot r vs z
    ax1.plot(z, r, '-*', **kwargs)
    ax1.set_xlabel(f"z [{unit}]", fontsize=fontsize)
    ax1.set_ylabel(f"r [{unit}]", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=minorsize)

    ## plot x vs y
    ax2.plot(x, y, '-*', **kwargs)
    ax2.set_xlabel(f"x [{unit}]", fontsize=fontsize)
    ax2.set_ylabel(f"y [{unit}]", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=minorsize)

    ## plot 3D
    ax3.plot3D(x, y, z, **kwargs)
    ax3.set_xlabel(f"x [{unit}]", fontsize=fontsize)
    ax3.set_ylabel(f"y [{unit}]", fontsize=fontsize)
    ax3.set_zlabel(f"z [{unit}]", fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=minorsize)
    ax3.scatter(x, y, z, **kwargs)
    
    ## plot r vs phi in polar coordinates
    phi = np.arctan2(y, x)
    ax4.plot(r, phi, **kwargs)
    ax4.set_xlabel(f"x [{unit}]", fontsize=fontsize)
    ax4.set_ylabel(f"y [{unit}]", fontsize=fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=minorsize)