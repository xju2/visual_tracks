"""This module compares tracking algorithms represented by two AthenaRawDataReader.
Each tracking algorithm is run in Athena frameework and produces a TrackCollection.
The two TrackCollections are dumped to text files and read by the reader.
 """
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acctrack.hparams_mixin import HyperparametersMixin

class TrackAlgComparator(HyperparametersMixin):
    def __init__(self, reader, other_reader, min_reco_clusters=5, name="TrackAlgComparator") -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["reader", "other_reader", "name"])
        self.name = name
        self.reader = reader
        self.other_reader = other_reader

        self.common_track_indices: List[Tuple[int, List[int], int]] = None
        self.common_info: pd.DataFrame = None
        self.common_track: pd.DataFrame = None
        self.common_other_track: pd.DataFrame = None

        self.unmatched_tracks: List[Tuple[int, List[int]]] = None
        self.disjoint_track_indices: List[Tuple[int, List[int]]] = None
        self.reverse_comparison = False
        self.redo_comparison = False


    def readers(self):
        return (self.reader, self.other_reader) \
                if not self.reverse_comparison \
                else (self.other_reader, self.reader)
    
    def reverse_comparison(self, do_reverse: bool=False) -> None:
        """Set the reverse comparison flag."""
        if self.reverse_comparison != do_reverse:
            # changing the comparison direction
            # so we need to redo the comparison
            self.redo_comparison = True
            self.reverse_comparison = do_reverse
        
    def compare_track_contents(self) -> Tuple[Any, Any, Any]:
        """Compares the tracks in the two TrackCollections. Returns a list of
        tuples (common_tracks, unmatched_tracks, disjoint_tracks) where
        common_tracks is a list of tuples (trkid, common_cluster_id, other_trkid) 
        unmatched_tracks is a list of tuples (trkids) not matched.
        """
            
        track_reader, other_track_reader = self.readers()
        min_num_clusters = self.hparams.min_reco_clusters

        label, other_label= track_reader.name, other_track_reader.name
        tracks, other_tracks = track_reader.tracks_clusters, other_track_reader.tracks_clusters

        num_matched = 0
        num_issubset = 0
        num_other_issubset = 0
        num_disjoints = 0

        unmatched_tracks = []
        disjoint_tracks = []
        common_tracks = []

        tot_tracks, tot_other_tracks = len(tracks), len(other_tracks)
        print(f"{tot_tracks} {label} tracks compared to {tot_other_tracks} {other_label} tracks.\n"
              f"Require min_num_clusters = {min_num_clusters} only for {label} tracks.")

        tot_filtered_tracks = 0
        for trkid,track in enumerate(tracks):
            if len(track) < min_num_clusters:
                continue
            tot_filtered_tracks += 1
            found_a_match = False
            all_disjoint = True
            found_as_subset = False
            found_other_as_subset = False
            ckf_idx = -1

            for idx,other_track in enumerate(other_tracks):
                track_set = set(track)
                other_track_set = set(other_track)
                if not track_set.isdisjoint(other_track_set):
                    all_disjoint = False
                if track_set == other_track_set:
                    found_a_match = True
                    ckf_idx = idx
                if track_set < other_track_set:
                    found_as_subset = True
                if track_set > other_track_set:
                    found_other_as_subset = True

            if found_a_match:
                num_matched += 1
                common_tracks.append((trkid,track,ckf_idx))
            else:
                unmatched_tracks.append((trkid,track))
            
            if all_disjoint:
                num_disjoints += 1
                disjoint_tracks.append((trkid,track))
            if found_as_subset:
                num_issubset += 1
            if found_other_as_subset:
                num_other_issubset += 1

        print(f"Total # of {label} tracks: {tot_tracks}. After filtering, # of {label} tracks: {tot_filtered_tracks} ({tot_filtered_tracks/tot_tracks*100:.3f}%)")
        print(f"Matched: {num_matched}, {tot_filtered_tracks}, {num_matched/tot_filtered_tracks:.4f}")
        print(f"{label} is a subset: {num_issubset}, {tot_filtered_tracks}, {num_issubset/tot_filtered_tracks:.4f}")
        print(f"{other_label} is a subset: {num_other_issubset}, {tot_filtered_tracks}, {num_other_issubset/tot_filtered_tracks:.4f}")
        print(f"Disjoint:  {num_disjoints}, {tot_filtered_tracks}, {num_disjoints/tot_filtered_tracks:.4f}")
        self.common_track_indices, self.unmatched_tracks, self.disjoint_track_indices = common_tracks, unmatched_tracks, disjoint_tracks

        # if we have already done the comparison, set the redo flag to False
        if self.redo_comparison:
            self.redo_comparison = False

        return (common_tracks, unmatched_tracks, disjoint_tracks)

    def analyse_common_track(self) -> pd.DataFrame:
        """Return a common track object."""
        if not self.redo_comparison and self.common_info is not None:
            return self.common_info
        
        if (self.redo_comparison or self.common_track_indices is None):
            self.compare_track_contents()
        
        common_tracks = self.common_track_indices
        reader, other_reader = self.readers()

        trk_id = np.array([x[0] for x in common_tracks])
        other_trk_id = np.array([x[2] for x in common_tracks])
        nclusters = np.array([len(x[1]) for x in common_tracks])
        df_common = pd.DataFrame({
            f"{reader.name}_trkid": trk_id,
            f"{other_reader.name}_trkid": other_trk_id,
            "nclusters": nclusters,
        })
        self.common_info = df_common

        self.common_track = reader.true_tracks[reader.true_tracks.trkid.isin(trk_id)]
        self.other_common_track = other_reader.true_tracks[
            other_reader.true_tracks.trkid.isin(other_trk_id)]
        
        return df_common


    def plot_common_tracks(self) -> Tuple[np.array, np.array]:
        """Analyze the common tracks. Compare their chi2 and other metrics."""
        df = self.analyse_common_track()
        reader, other_reader = self.readers()
        label, other_label = reader.name, other_reader.name
        
        num_common_tracks = len(df)

        # number of clusters
        plt.title("Common tracks")
        plt.hist(df.nclusters.values, bins=31, range=(-0.5,30.5),
                 label=f"Total {num_common_tracks}", alpha=0.5)
        plt.legend()
        plt.xlabel("number of clusters")
        plt.show()

        # chi2 / ndof
        chi2_hist_config = dict(bins=50, range=(0, 4),
                                alpha=0.5, histtype='step', lw=2)
        chi2 = self.common_track.chi2.values / self.common_track.nDoF.values
        other_chi2 = self.other_common_track.chi2.values / self.other_common_track.nDoF.values
        plt.title("Common Tracks")
        plt.hist(chi2, **chi2_hist_config, label=label)
        plt.hist(other_chi2, **chi2_hist_config, label=other_label)
        plt.xlim(0, 4)
        plt.xlabel("$\chi^2$/ndof")
        plt.legend()
        plt.show()

        # scatter plot for chi2 / ndof
        plt.title("Common Tracks")
        config = dict(s=10, alpha=0.5)
        plt.scatter(chi2, other_chi2, **config)
        plt.plot([0, 4], [0, 4], color='red', linestyle='--')
        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.xlabel(f"{label} $\chi^2$/ndof")
        plt.ylabel(f"{other_label} $\chi^2$/ndof")
        plt.show()

        # difference in chi2 / ndof
        delta_chi2 = chi2 - other_chi2
        plt.title("Common Tracks")
        bin_values, _, _ = plt.hist(delta_chi2, bins=50, range=(-2, 2), alpha=0.5)
        max_bin_value, min_bin_value = np.max(bin_values), np.min(bin_values)
        y_start = max_bin_value*0.5
        delta = (max_bin_value - min_bin_value) * 0.08
        plt.text(-1.5, y_start, f"Mean: {np.mean(delta_chi2):8.4f}", fontsize=12)
        plt.text(-1.5, y_start - delta, f"Std:  {np.std(delta_chi2):8.4f}", fontsize=12)
        plt.xlabel(f"({label} - {other_label}) $\chi^2$/ndof")
        plt.plot([0, 0], [0, max_bin_value], color='red', linestyle='--')
        plt.show()

        return chi2, other_chi2