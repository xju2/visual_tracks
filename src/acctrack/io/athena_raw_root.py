from __future__ import annotations

from pathlib import Path

import awkward
import numpy as np
import pandas as pd
import uproot

from acctrack.io import utils_athena_raw as utils_raw_csv
from acctrack.io import utils_athena_raw_root as utils_raw_root
from acctrack.io.base import BaseTrackDataReader


class AthenaRawRootReader(BaseTrackDataReader):
    """Read Raw ROOT files created from RDO files with the dumping file.

    The code that creates the ROOT file can be found
    [link](https://gitlab.cern.ch/atlas/athena/-/tree/main/InnerDetector/InDetGNNTracking?ref_type=heads)
    """

    def __init__(
        self, inputdir, output_dir=None, overwrite=True, name="AthenaRawRootReader"
    ):
        super().__init__(inputdir, output_dir, overwrite, name)

        # find all files in inputdir
        root_files = sorted(self.inputdir.glob("*.root"))

        self.tree_name = "GNN4ITk"

        # create a map: event number -> file index and entry index
        self.event_map: dict[int, tuple[int, int]] = {}
        file_idx = 0
        self.root_files = []
        self.tot_evts = 0
        for filename in root_files:
            file_handle = uproot.open(filename)
            if self.tree_name not in file_handle:
                print(f"Tree {self.tree_name} is not in {filename}")
                continue

            events = uproot.open(f"{filename}:{self.tree_name}")
            event_numbers = events.arrays(["event_number"])["event_number"]
            num_entries = len(event_numbers)
            self.event_map.update(
                {
                    event_number: (file_idx, entry)
                    for event_number, entry in zip(event_numbers, range(num_entries))
                }
            )
            self.root_files.append(filename)
            self.tot_evts += num_entries
            file_idx += 1

        self.num_files = len(self.root_files)
        print(
            f"{self.inputdir} contains  {self.num_files} files and total {self.tot_evts} events."
        )

    def read_file(self, file_idx: int = 0, max_evts: int = -1) -> list[int]:
        """Read all events from the ROOT file. Each event is saved in parquet format.

        Args
        ----
            file_idx: int, index of the ROOT file to read in the input directory.
            max_evts: int, maximum number of events to read. If -1, read all events.

        Returns
        -------
            list of event numbers: list[int]. Return None if file_idx is out of range.
        """
        if file_idx >= self.num_files:
            print(
                f"File index {file_idx} is out of range. Max index is {self.num_files - 1}"
            )
            return None
        filename = self.root_files[file_idx]
        print(f"Reading file: {filename}")
        tree = uproot.open(f"{filename}:{self.tree_name}")

        all_event_info = tree.arrays(utils_raw_root.all_branches)
        num_events = len(all_event_info["event_number"])
        print(f"Number of events: {num_events} in file {filename}.")
        if max_evts > 0 and max_evts < num_events:
            print(f"Reading only {max_evts} events.")
            num_events = max_evts

        event_numbers = [
            self.process_one_event(all_event_info[evtid]) for evtid in range(num_events)
        ]
        return event_numbers

    def process_one_event(self, tracks_info: awkward.highlevel.Array) -> int:
        """Process one event and save the data in parquet format.

        Args
        ----
            tracks_info: awkward.highlevel.Array, the data of one event.

        Returns
        -------
            event_number: int, the event number of the processed event.
        """
        event_number = tracks_info["event_number"]

        # read particles
        particle_arrays = [tracks_info[x] for x in utils_raw_root.particle_branch_names]
        particles = pd.DataFrame(
            dict(zip(utils_raw_root.particle_col_names, particle_arrays))
        )
        # convert barcode to 7 digits
        particle_ids = utils_raw_csv.get_particle_ids(particles)
        particles.insert(0, "particle_id", particle_ids)
        particles = utils_raw_csv.particles_of_interest(particles)
        self._save("particles", particles, event_number)

        # read clusters
        cluster_arrays = [
            tracks_info[x]
            for x in utils_raw_root.cluster_branch_names
            if x
            not in {"CLhardware", "CLparticleLink_barcode", "CLparticleLink_eventIndex"}
        ]
        # hardware is a std::vector, need special treatment
        cluster_hardware = np.array(tracks_info["CLhardware"].tolist(), dtype=str)
        cluster_columns = [*utils_raw_root.cluster_col_names, "hardware"]
        cluster_arrays.append(cluster_hardware)
        clusters = pd.DataFrame(dict(zip(cluster_columns, cluster_arrays)))

        clusters = clusters.astype({"hardware": "str", "barrel_endcap": "int32"})
        # read truth links for each cluster
        subevent_name, barcode_name = utils_raw_root.cluster_link_branch_names
        matched_subevents = tracks_info[subevent_name].tolist()
        matched_barcodes = tracks_info[barcode_name].tolist()
        max_matched = max(len(x) for x in matched_subevents)

        # loop over clusters matched particles
        matched_info = []
        for entry_idx in range(max_matched):
            matched_info += [
                (cluster_id, subevent[entry_idx], barcode[entry_idx])
                for cluster_id, subevent, barcode in zip(
                    clusters["cluster_id"].values,
                    matched_subevents,
                    matched_barcodes,
                )
                if len(subevent) > entry_idx
            ]
        cluster_matched = pd.DataFrame(
            matched_info, columns=["cluster_id", "subevent", "barcode"]
        )
        cluster_matched["particle_id"] = utils_raw_csv.get_particle_ids(cluster_matched)
        clusters = clusters.merge(cluster_matched, on="cluster_id", how="left")

        self._save("clusters", clusters, event_number)

        # read spacepoints
        spacepoint_arrays = [
            tracks_info[x] for x in utils_raw_root.spacepoint_branch_names
        ]
        spacepoints = pd.DataFrame(
            dict(zip(utils_raw_root.spacepoint_col_names, spacepoint_arrays))
        )
        self._save("spacepoints", spacepoints, event_number)

        pixel_hits = spacepoints[spacepoints["cluster_index_2"] == -1]
        strip_hits = spacepoints[spacepoints["cluster_index_2"] != -1]

        # matching spacepoints to particles through clusters
        truth = utils_raw_csv.truth_match_clusters(pixel_hits, strip_hits, clusters)
        # regional labels
        # region_labels = {
        #     1: {"hardware": "PIXEL", "barrel_endcap": -2},
        #     2: {"hardware": "STRIP", "barrel_endcap": -2},
        #     3: {"hardware": "PIXEL", "barrel_endcap": 0},
        #     4: {"hardware": "STRIP", "barrel_endcap": 0},
        #     5: {"hardware": "PIXEL", "barrel_endcap": 2},
        #     6: {"hardware": "STRIP", "barrel_endcap": 2},
        # }
        truth = utils_raw_csv.merge_spacepoints_clusters(truth, clusters)
        # truth = utils_raw_csv.add_region_labels(truth, region_labels)
        self._save("truth", truth, event_number)

        # read detailed truth tracks (dtt)
        dtt_arrays = [
            tracks_info[x].to_numpy()
            for x in utils_raw_root.detailed_truth_branch_names
            if "trajectory" not in x
        ]
        detailed_matching = pd.DataFrame(
            np.array(
                [
                    dtt_arrays[0],
                    dtt_arrays[1],
                    dtt_arrays[2][:, 0],
                    dtt_arrays[2][:, 1],
                    dtt_arrays[3][:, 0],
                    dtt_arrays[3][:, 1],
                    dtt_arrays[4][:, 0],
                    dtt_arrays[4][:, 1],
                ]
            ).T,
            columns=[
                "trkid",
                "num_matched",
                "true_pixel_hits",
                "true_sct_hits",
                "reco_pixel_hits",
                "reco_sct_hits",
                "common_pixel_hits",
                "common_sct_hits",
            ],
        )
        self._save("detailed_matching", detailed_matching, event_number)

        return event_number

    def _save(self, outname: str, df: pd.DataFrame, evtid: int) -> bool:
        outname = self.get_outname(outname, evtid)
        if outname.exists() and not self.overwrite:
            return True
        if df is not None:
            df.to_parquet(outname, compression="gzip")
        return False

    def _read(self, outname: str, evtid: int) -> pd.DataFrame:
        outname = self.get_outname(outname, evtid)
        if not outname.exists():
            return None
        return pd.read_parquet(outname)

    def get_outname(self, outname: str, evtid: int) -> Path:
        return self.outdir / f"event{evtid:06d}-{outname}.parquet"

    def read(self, evtid: int = 0) -> bool:
        self.clusters = self._read("clusters", evtid)
        self.particles = self._read("particles", evtid)
        self.spacepoints = self._read("spacepoints", evtid)
        self.truth = self._read("truth", evtid)
        if any(
            x is None
            for x in [self.clusters, self.particles, self.spacepoints, self.truth]
        ):
            print(f"event {evtid} are not processed.")
            print("please run `read_file()` first!")
            return False
        return True

    def get_event_info(self, file_idx: int = 0) -> pd.DataFrame:
        if file_idx >= self.num_files:
            print(
                f"File index {file_idx} is out of range. Max index is {self.num_files - 1}"
            )
            return None

        filename = self.root_files[file_idx]
        print("reading event info from", filename)
        with uproot.open(filename) as f:
            tree = f[self.tree_name]
            event_info = tree.arrays(utils_raw_root.event_branch_names, library="pd")
            return event_info

    def find_event(self, event_numbers: list[int]):
        # we loop over all availabel root files
        # check if the requrested event number is in the file
        # if yes, we write down the file name and the event number
        # if no, we continue to the next file

        # event_number_map: Dict[int, str] = dict([(x, "") for x in event_numbers])
        event_number_map: dict[int, str] = {}

        for root_file_idx in range(len(self.root_files)):
            event_info = self.get_event_info(root_file_idx)
            for event_number in event_numbers:
                if event_number in event_info["event_number"].to_numpy():
                    print(
                        f"Event {event_number} is in file {self.root_files[root_file_idx]}"
                    )
                    event_number_map[event_number] = self.root_files[root_file_idx]

                    self.read_file(root_file_idx)
                    self.truth.to_csv(f"event{event_number:09d}-truth.csv")

        return event_number_map

    def match_to_truth(self) -> pd.DataFrame:
        """Match a reco track to a truth track
        only if all reco track contents are from the truth track.
        """
        detailed = self.detailed_matching
        all_matched_to_truth = detailed[
            (detailed.reco_pixel_hits == detailed.common_pixel_hits)
            & (detailed.reco_sct_hits == detailed.common_sct_hits)
        ]
        num_all_matched_to_truth = len(all_matched_to_truth)

        frac_all_matched_to_truth = num_all_matched_to_truth / len(detailed)
        print(
            f"All matched to truth {self.name}: ",
            num_all_matched_to_truth,
            len(detailed),
            frac_all_matched_to_truth,
        )

        # Check that each reco track is matched to only one truth track
        _, counts = np.unique(all_matched_to_truth.trkid.values, return_counts=True)
        assert np.all(
            counts == 1
        ), "Some reco tracks are matched to multiple truth tracks"

        self.tracks_matched_to_truth = all_matched_to_truth

        return all_matched_to_truth
