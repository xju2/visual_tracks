"""Visualize tracks."""

import numpy as np
import matplotlib.pyplot as plt


def plot(track: np.ndarray,
        figsize=(15,12),
        fontsize=16,
        minorsize=14,
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
    ax1.plot(z, r, **kwargs)
    ax1.set_xlabel("z [m]", fontsize=fontsize)
    ax1.set_ylabel("r [m]", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=minorsize)

    ## plot x vs y
    ax2.plot(x, y, **kwargs)
    ax2.set_xlabel("x [m]", fontsize=fontsize)
    ax2.set_ylabel("y [m]", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=minorsize)

    ## plot 3D
    ax3.plot3D(x, y, z, **kwargs)
    ax3.set_xlabel("x [m]", fontsize=fontsize)
    ax3.set_ylabel("y [m]", fontsize=fontsize)
    ax3.set_zlabel("z [m]", fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=minorsize)
    ax3.scatter(x, y, z, **kwargs)
    
    ## plot x vs y in polar coordinates
    ax4.plot(x, y, **kwargs)
    ax4.set_xlabel("x [m]", fontsize=fontsize)
    ax4.set_ylabel("y [m]", fontsize=fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=minorsize)