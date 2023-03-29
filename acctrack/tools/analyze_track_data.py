"""Analyze the track data handled by a data reader"""
from acctrack.utils.utils_plot import add_mean_std, create_figure
from acctrack.io import BaseTrackDataReader

import matplotlib.pyplot as plt

class AnalyseTrackData:
    def __init__(self, reader: BaseTrackDataReader):
        self.reader = reader

    def study_cluster_features(self, evtid: int = 0):
        """Study the cluster features"""
        self.reader.read(evtid)
        if self.clusters is None:
            print(f"No cluster information found in {self.reader.name} for event {evtid}.")

        clusters = self.reader.clusters
        print("Number of clusters: ", clusters.shape[0])
        pixel_clusters = clusters[clusters["hardware"] == "PIXEL"]
        strip_clusters = clusters[clusters["hardware"] == "STRIP"]
        print("Number of pixel clusters: ", pixel_clusters.shape[0])
        print("Number of strip clusters: ", strip_clusters.shape[0])

        # number of pixels per cluster
        fig, ax = create_figure()
        config = dict(bins=31, range=(-0.5, 30.5), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["pixel_count"], label="Pixel clusters", **config)
        ax.hist(strip_clusters["pixel_count"], label="Strip clusters", **config)
        ax.set_xlabel("# of Pixels per cluster")
        plt.legend()
        plt.show()

        # number of charges per cluster
        print("Charge information is not available for strip clusters")
        _, ax = create_figure()
        config = dict(bins=51, range=(-0.5, 50.5), histtype="step", lw=2)
        ax.hist(pixel_clusters["charge_count"], label="Pixel clusters", **config)
        ax.set_xlabel("# of Charges per cluster")
        add_mean_std(pixel_clusters["charge_count"], 15, 15000, ax, dy=2500)
        plt.legend()
        plt.show()

        # cluster position
        _, ax = create_figure()
        config = dict(bins=600, range=(-3000, 3000), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["cluster_x"], label="pixel x", **config)
        ax.hist(pixel_clusters["cluster_y"], label="pixel y", **config)
        ax.hist(pixel_clusters["cluster_z"], label="pixel z", **config)
        ax.hist(strip_clusters["cluster_x"], label="strip x", **config, linestyle="--")
        ax.hist(strip_clusters["cluster_y"], label="strip y", **config, linestyle="--")
        ax.hist(strip_clusters["cluster_z"], label="strip z", **config, linestyle="--")
        ax.set_xlabel("Cluster position [mm]")
        plt.legend()
        plt.show()

        # local directions of clusters
        _, ax = create_figure()
        config = dict(bins=50, range=(0, 30.0), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["loc_direction1"], label="pixel x", **config)
        ax.hist(pixel_clusters["loc_direction2"], label="pixel y", **config)
        ax.hist(pixel_clusters["loc_direction3"], label="pixel z", **config)
        ax.hist(strip_clusters["loc_direction1"], label="strip x", **config, linestyle="--")
        ax.hist(strip_clusters["loc_direction2"], label="strip y", **config, linestyle="--")
        ax.hist(strip_clusters["loc_direction3"], label="strip z", **config, linestyle="--")
        ax.set_xlabel("Local directions")
        plt.legend()
        plt.show()

        # eta/phi from local/global directions of clusters
        _, ax = create_figure()
        config = dict(bins=50, range=(-3.15, 3.15), histtype="step", lw=2, alpha=0.8)
        plt.title("Pixel clusters"")
        ax.hist(pixel_clusters["loc_eta"], label=r"local $\eta$", **config)
        ax.hist(pixel_clusters["loc_phi"], label=r"local $\phi$", **config)
        ax.hist(pixel_clusters["glob_eta"], label=r"global $\eta$", **config, linestyle="--")
        ax.hist(pixel_clusters["glob_phi"], label=r"global $\phi$", **config, linestyle="--")
        ax.set_xlabel("Cluster Shapes")
        plt.legend()
        plt.show()
