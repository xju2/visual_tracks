"""Analyze the track data handled by a data reader"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acctrack.utils.utils_plot import add_mean_std, create_figure
from acctrack.io import BaseTrackDataReader

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
        plt.title("Pixel clusters")
        ax.hist(pixel_clusters["loc_eta"], label=r"local $\eta$", **config)
        ax.hist(pixel_clusters["loc_phi"], label=r"local $\phi$", **config)
        ax.hist(pixel_clusters["glob_eta"], label=r"global $\eta$", **config, linestyle="--")
        ax.hist(pixel_clusters["glob_phi"], label=r"global $\phi$", **config, linestyle="--")
        ax.set_xlabel("Cluster Shapes")
        plt.legend()
        plt.show()

    def apply_eta_dep_cuts(self,
                           track: pd.DataFrame,
                           eta_bins: List[float] = [0, 2.0, 2.6, 4.0],
                           min_hits: List[int] = [9, 8, 7],
                           min_pT: List[float] = [900, 400, 400],  # MeV
                           max_oot: List[float] = [10] * 3,
                           chi2_ndof: List[float] = [7] * 3,
                           ):
        """Apply eta dependent cuts.
        The default cuts are taken the ATLAS ITk performance paper.

        eta: eta of the track
        pT: pT of the track
        mot: measurements on track
        oot: outliers on track
        """
        required_features = ["eta", "pt", "mot", "oot", "chi2_ndof"]
        assert all([f in track.columns for f in required_features]), \
            f"Track data does not contain all required features: {required_features}"

        # apply eta dependent cuts on number of hits
        track = track.assign(abseta=track["eta"].abs())

        def apply_cuts(value: str, cut_list: List[float], cmp_opt: str = ">"):
            query = "&".join([
                f"(abseta > {eta_bins[idx]}) & (abseta <= {eta_bins[idx+1]}) & ({value} {cmp_opt} {cut})"
                for idx, cut in enumerate(cut_list)])
            return track.query(query)

        num_hits_cuts = apply_cuts("mot", min_hits, "<")
        pt_cuts = apply_cuts("pt", min_pT, "<=")
        outlier_cuts = apply_cuts("oot", max_oot, ">")
        chi2_cuts = apply_cuts("chi2ndo", chi2_ndof, ">")

        num_failed_hits = track[num_hits_cuts].shape[0]
        num_failed_pt = track[pt_cuts].shape[0]
        num_failed_outliers = track[outlier_cuts].shape[0]
        num_failed_chi2 = track[chi2_cuts].shape[0]
        num_failed_all = track[num_hits_cuts | pt_cuts | outlier_cuts | chi2_cuts].shape[0]
        print("Total number of tracks: ", track.shape[0])
        print("Number of tracks failing number of hits cut: ",
              num_failed_hits, f"({num_failed_hits/track.shape[0]*100:.2f}%)")
        print("Number of tracks failing pT cut: ",
              num_failed_pt, f"({num_failed_pt/track.shape[0]*100:.2f}%)")
        print("Number of tracks failing outlier cut: ",
              num_failed_outliers, f"({num_failed_outliers/track.shape[0]*100:.2f}%)")
        print("Number of tracks failing chi2 cut: ",
              num_failed_chi2, f"({num_failed_chi2/track.shape[0]*100:.2f}%)")
        print("Number of tracks failing all cuts: ",
              num_failed_all, f"({num_failed_all/track.shape[0]*100:.2f}%)")

        # number of hits vs |eta|
        _, ax = create_figure()
        ax.scatter(track.abseta, track.mot, alpha=0.5, s=10)

        def add_cut_lines(ax, cut_list: List[float]):
            if len(eta_bins) > 1:
                for idx, cut in enumerate(cut_list):
                    ax.plot([eta_bins[idx], eta_bins[idx + 1]], [cut, cut], color="red")
                for idx, cut in range(1, len(eta_bins) - 1):
                    ax.plot([eta_bins[idx], eta_bins[idx]], [cut[idx - 1], cut[idx]], color="red")
            else:
                plt.axhline(cut_list[0], color="red")

        add_cut_lines(ax, min_hits)
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel("# of clusters")
        plt.show()

        # pT vs |eta|
        _, ax = create_figure()
        ax.scatter(track.abseta, track.pt, alpha=0.5, s=10)
        add_cut_lines(ax, min_pT)
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel("pT [MeV]")
        ax.set_ylim(0, 5000)
        plt.show()

        return track[num_hits_cuts | pt_cuts | outlier_cuts | chi2_cuts]
