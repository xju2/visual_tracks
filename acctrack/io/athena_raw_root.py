from typing import Dict, Tuple, List
from pathlib import Path

import json
import uproot

import pandas as pd
import numpy as np

from acctrack.io import utils_athena_raw_root as utils_raw_root
from acctrack.io import utils_athena_raw as utils_raw_csv
from acctrack.io.base import BaseTrackDataReader


class AthenaRawRootReader(BaseTrackDataReader):
    def __init__(self, inputdir, output_dir=None,
                 overwrite=True, name="AthenaRawRootReader"):
        super().__init__(inputdir, output_dir, overwrite, name)

        # find all files in inputdir
        self.root_files = sorted(list(self.inputdir.glob("*.root")))

        self.tree_name = "GNN4ITk"

        # now we read all files and determine the starting event id for each file
        self.file_evtid = [0]
        for filename in self.root_files:
            try:
                num_entries = list(uproot.num_entries(str(filename) + ":" + self.tree_name))[0][-1]
            except OSError:
                print(f"Error reading file: {filename}")
                self.root_files.remove(filename)
                continue

            start_evtid = self.file_evtid[-1] + num_entries
            self.file_evtid.append(start_evtid)

        self.num_files = len(self.root_files)
        print(f"{self.inputdir} contains  {self.num_files} files and total {self.file_evtid[-1]} events.")

    def read_file(self, file_idx: int = 0) -> uproot.models.TTree:
        if file_idx >= self.num_files:
            print(f"File index {file_idx} is out of range. Max index is {self.num_files - 1}")
            return None
        filename = self.root_files[file_idx]
        print(f"Reading file: {filename}")
        file = uproot.open(filename)
        tree = file[self.tree_name]
        evtid = self.file_evtid[file_idx]

        self.tree = tree
        file_map: Dict[int, Tuple[int, int]] = {}
        out_filenames = ["particles", "clusters", "spacepoints", "truth"]
        for batch in tree.iterate(step_size=1, filter_name=utils_raw_root.all_branches, library="np"):
            # the index 0 is because we have step_size = 1
            # read event info
            run_number = batch["run_number"][0]
            event_number = batch["event_number"][0]
            file_map[evtid] = (run_number, event_number)

            # check if all files exists. If yes, skip the event
            is_file_exist = [self._save(x, None, evtid) for x in out_filenames]
            if not self.overwrite and all(is_file_exist):
                evtid += 1
                continue

            # read particles
            particle_arrays = [batch[x][0] for x in utils_raw_root.particle_branch_names]
            particles = pd.DataFrame(dict(zip(utils_raw_root.particle_columns, particle_arrays)))
            particles = particles.rename(columns={"event_number": "subevent"})
            # convert barcode to 7 digits
            particle_ids = utils_raw_csv.get_particle_ids(particles)
            particles.insert(0, "particle_id", particle_ids)
            particles = utils_raw_csv.particles_of_interest(particles)
            self._save("particles", particles, evtid)


            # read clusters
            cluster_arrays = [batch[x][0] for x in utils_raw_root.cluster_branch_names]
            # hardware is a std::vector, need special treatment
            cluster_hardware = np.array(batch["CLhardware"][0].tolist(), dtype=np.str)
            cluster_columns = utils_raw_root.cluster_columns + ["hardware"]
            cluster_arrays.append(cluster_hardware)
            clusters = pd.DataFrame(dict(zip(cluster_columns, cluster_arrays)))
            clusters = clusters.rename(columns={
                "index": "cluster_id",
                "x": "cluster_x",
                "y": "cluster_y",
                "z": "cluster_z",
                "pixel_count": "count",
                "loc_eta": "leta",
                "loc_phi": "lphi",
                "loc_direction1": "localDir0",
                "loc_direction2": "localDir1",
                "loc_direction3": "localDir2",
                "Jan_loc_direction1": "lengthDir0",
                "Jan_loc_direction2": "lengthDir1",
                "Jan_loc_direction3": "lengthDir2",
                "moduleID": "module_id"   # <TODO, not a correct module id>
            })
            clusters['cluster_id'] = clusters['cluster_id'] - 1
            clusters = clusters.astype({"hardware": "str", "barrel_endcap": "int32"})
            # read truth links for each cluster
            subevent_name, barcode_name = utils_raw_root.cluster_link_branch_names
            matched_subevents = batch[subevent_name][0].tolist()
            matched_barcodes = batch[barcode_name][0].tolist()
            max_matched = max([len(x) for x in matched_subevents])
            # loop over clusters matched particles
            matched_info = []
            for idx in range(max_matched):
                matched_info += [
                    (cluster_id, subevent[idx], barcode[idx]) for cluster_id, subevent, barcode in zip(
                        clusters["cluster_id"].values, matched_subevents, matched_barcodes)
                    if len(subevent) > idx
                ]
            cluster_matched = pd.DataFrame(matched_info, columns=["cluster_id", "subevent", "barcode"])
            cluster_matched["particle_id"] = utils_raw_csv.get_particle_ids(cluster_matched)
            clusters = clusters.merge(cluster_matched, on="cluster_id", how="left")

            self._save("clusters", clusters, evtid)


            # read spacepoints
            spacepoint_arrays = [batch[x][0] for x in utils_raw_root.spacepoint_branch_names]
            spacepoints = pd.DataFrame(dict(zip(utils_raw_root.spacepoint_columns, spacepoint_arrays)))
            spacepoints = spacepoints.rename(columns={
                "index": "hit_id", "CL1_index": "cluster_index_1", "CL2_index": "cluster_index_2"
            })
            self._save("spacepoints", spacepoints, evtid)

            pixel_hits = spacepoints[spacepoints["cluster_index_2"] == -1]
            strip_hits = spacepoints[spacepoints["cluster_index_2"] != -1]


            # matching spacepoints to particles through clusters
            truth = utils_raw_csv.truth_match_clusters(pixel_hits, strip_hits, clusters)
            # regional labels
            region_labels = dict([(1, {"hardware": "PIXEL", "barrel_endcap": -2}),
                                  (2, {"hardware": "STRIP", "barrel_endcap": -2}),
                                  (3, {"hardware": "PIXEL", "barrel_endcap": 0}),
                                  (4, {"hardware": "STRIP", "barrel_endcap": 0}),
                                  (5, {"hardware": "PIXEL", "barrel_endcap": 2}),
                                  (6, {"hardware": "STRIP", "barrel_endcap": 2})
                                  ])
            truth = utils_raw_csv.merge_spacepoints_clusters(truth, clusters)
            truth = utils_raw_csv.add_region_labels(truth, region_labels)
            self._save("truth", truth, evtid)
            evtid += 1

        # save file map
        with open(self.outdir / f"filemap_${file_idx}.json", "w") as f:
            json.dump(file_map, f)
        return tree

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
        if any([x is None for x in [
                self.clusters, self.particles, self.spacepoints, self.truth]]):
            print("event {evtid} are not processed.")
            print("please run `read_file()` first!")
            return False
        else:
            return True

    def get_event_info(self, file_idx: int = 0) -> pd.DataFrame:
        if file_idx >= self.num_files:
            print(f"File index {file_idx} is out of range. Max index is {self.num_files - 1}")
            return None

        filename = self.root_files[file_idx]
        print("reading event info from", filename)
        with uproot.open(filename) as f:
            tree = f[self.tree_name]
            event_info = tree.arrays(utils_raw_root.event_branch_names, library="pd")
            return event_info

    def find_event(self, event_numbers: List[int]):
        # we loop over all availabel root files
        # check if the requrested event number is in the file
        # if yes, we write down the file name and the event number
        # if no, we continue to the next file

        # event_number_map: Dict[int, str] = dict([(x, "") for x in event_numbers])
        event_number_map: Dict[int, str] = {}

        for root_file_idx in range(len(self.root_files)):
            event_info = self.get_event_info(root_file_idx)
            for event_number in event_numbers:
                if event_number in event_info["event_number"].values:
                    print(f"Event {event_number} is in file {self.root_files[root_file_idx]}")
                    event_number_map[event_number] = self.root_files[root_file_idx]

        return event_number_map
