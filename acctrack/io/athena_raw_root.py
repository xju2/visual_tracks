from pathlib import Path
import uproot

import pandas as pd

from acctrack.io import utils_athena_raw_root as utils
from acctrack.io import utils as io_utils

class AthenaRawRootReader:
    def __init__(self, inputdir, output_dir=None, overwrite=False, name="AthenaRawRootReader"):
        self.inputdir = Path(inputdir)
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(f"Input directory {self.inputdir} does not exist or is not a directory.")

        self.outdir = Path(output_dir) if output_dir else self.inputdir / "processed_data"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.overwrite = overwrite
        self.global_evtid = 0

        # find all files in inputdir
        self.root_files = list(self.inputdir.glob("*.root"))
        self.num_files = len(self.root_files)
        print(f"Total number of files: {self.num_files} in {self.inputdir}")

    def read(self, file_idx: int = 0):
        if file_idx >= self.num_files:
            print(f"File index {file_idx} is out of range. Max index is {self.num_files - 1}")
            print("Read the first file.")
            file_idx = 0
        filename = self.root_files[file_idx]
        file = uproot.open(filename)
        tree_name = "GNN4ITk"
        tree = file[tree_name]
        num_entries = tree.num_entries
        print(f"{num_entries} entries of tree {tree_name} in {filename}")

        self.tree = tree
        self.num_entries = num_entries
        return tree

    def read_particles(self):
        arrays = utils.create_arrays(
            self.tree.arrays(utils.particle_branch_names, library="np")
        )
        df = pd.DataFrame(arrays, columns=utils.particle_columns)
        return df

