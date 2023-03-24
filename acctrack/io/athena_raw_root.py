from pathlib import Path
import uproot

import pandas as pd
import numpy as np

from acctrack.io import utils_athena_raw_root as utils_raw_root
from acctrack.io import utils_athena_raw as utils_raw_csv


class AthenaRawRootReader:
    def __init__(self, inputdir, output_dir=None, overwrite=True, name="AthenaRawRootReader"):
        self.inputdir = Path(inputdir)
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(f"Input directory {self.inputdir} does not exist or is not a directory.")

        self.outdir = Path(output_dir) if output_dir else self.inputdir / "processed_data"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.overwrite = overwrite
        self.global_evtid = 0
        self.tree_name = "GNN4ITk"


        # find all files in inputdir
        self.root_files = list(self.inputdir.glob("*.root"))
        self.num_files = len(self.root_files)

        # now we read all files and determine the starting event id for each file
        self.file_evtid = [0]
        for filename in self.root_files:
            num_entries = list(uproot.num_entries(str(filename) + ":" + self.tree_name))[0][-1]
            start_evtid = sum(self.file_evtid) + num_entries
            self.file_evtid.append(start_evtid)
        print(f"{self.inputdir} contains  {self.num_files} files and total {self.file_evtid[-1]} events.")

    def read_file(self, file_idx: int = 0) -> uproot.models.TTree:
        if file_idx >= self.num_files:
            print(f"File index {file_idx} is out of range. Max index is {self.num_files - 1}")
            return None
        filename = self.root_files[file_idx]
        file = uproot.open(filename)
        tree_name = "GNN4ITk"
        tree = file[tree_name]
        self.global_evtid = self.file_evtid[file_idx]

        self.tree = tree
        for batch in tree.iterate(step_size=1, filter_name=utils_raw_root.all_branches, library="np"):
            # the index 0 is because we have step_size = 1

            # read particles
            particle_arrays = [batch[x][0] for x in utils_raw_root.particle_branch_names]
            particles = pd.DataFrame(dict(zip(utils_raw_root.particle_columns, particle_arrays)))
            particles = particles.rename(columns={"event_number": "subevent"})
            # convert barcode to 7 digits
            particle_ids = utils_raw_csv.get_particle_ids(particles)
            particles.insert(0, "particle_id", particle_ids)

            self._save("particles", particles)


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
                "z": "cluster_z"
            })
            clusters['cluster_id'] = clusters['cluster_id'] - 1
            clusters = clusters.astype({"hardware": "str", "barrel_endcap": "int32"})
            # read truth links for each cluster
            cluster_match = []
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
            self._save("clusters", clusters)


            # read spacepoints
            spacepoint_arrays = [batch[x][0] for x in utils_raw_root.spacepoint_branch_names]
            spacepoints = pd.DataFrame(dict(zip(utils_raw_root.spacepoint_columns, spacepoint_arrays)))
            spacepoints = spacepoints.rename(columns={
                "index": "hit_id", "CL1_index": "cluster_index_1", "CL2_index": "cluster_index_2"
            })
            self._save("spacepoints", spacepoints)

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
            self._save("truth", truth)
            break

            self.global_evtid += 1
        return tree

    def _save(self, outname: str, df: pd.DataFrame):
        outname = self.get_outname(outname)
        if outname.exists() and not self.overwrite:
            return
        df.to_parquet(outname, compression="gzip")

    def _read(self, outname: str) -> pd.DataFrame:
        outname = self.get_outname(outname)
        if not outname.exists():
            return None
        return pd.read_parquet(outname)

    def get_outname(self, outname: str) -> Path:
        return self.outdir / f"event{self.global_evtid:06d}-{outname}.parquet"

    def read(self, evtid: int = 0):
        self.global_evtid = evtid
        self.clusters = self._read("clusters")
        self.particles = self._read("particles")
        self.spacepoints = self._read("spacepoints")
        self.truth = self._read("truth")
