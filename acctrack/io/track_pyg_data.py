"""This moudle reads the PyG data object created by the CommomFramework.
"""
import re

import torch
from acctrack.io.base import BaseTrackDataReader

class TrackGraphDataReader(BaseTrackDataReader):
    def __init__(self, inputdir: str, output_dir: str = None,
                 overwrite: bool = True, name="BaseTrackDataReader"):
        super().__init__(inputdir, output_dir, overwrite, name)

        # find all files in inputdir
        self.pyg_files = list(self.inputdir.glob("*.pyg"))
        self.nevts = len(self.pyg_files)

        # get event ids.
        pattern = "event(.*).pyg"
        regrex = re.compile(pattern)

        def find_evt_info(x):
            matched = regrex.search(x.name)
            if matched is None:
                return None
            evtid = int(matched.group(1).strip())
            return evtid

        self.all_evtids = sorted([find_evt_info(x) for x in self.pyg_files])
        print("Total {} events in directory: {}".format(
            self.nevts, self.basedir))

    def read(self, evtid: int = 0) -> bool:
        """Read one event from the input directory."""
        filename = self.pyg_files[evtid]
        print("Reading file: {}".format(filename))
        data = torch.load(filename, map_location='cpu')
        return data