"""This moudle reads the PyG data object created by the CommomFramework.
"""
from typing import Union, List, Optional
import re
from pathlib import Path

import torch
from acctrack.io.base import BaseTrackDataReader

class TrackGraphDataReader(BaseTrackDataReader):
    def __init__(self, inputdir: Union[str, Path], output_dir: str = None,
                 overwrite: bool = True, name="BaseTrackDataReader"):
        super().__init__(inputdir, output_dir, overwrite, name)

        # find all files in inputdir
        self.pyg_files = list(self.inputdir.glob("*.pyg"))
        self.nevts = len(self.pyg_files)

        # get event ids.
        # pattern = "event\[\[(.*)\]\].pyg"
        regrex = re.compile("event([0-9]*).pyg")

        def find_evt_info(x):
            matched = regrex.search(x.name)
            if matched is None:
                return None
            evtid = int(matched.group(1).strip("'").strip("0"))
            return evtid

        self.all_evtids = sorted([find_evt_info(x) for x in self.pyg_files])
        print("{}: Total {} events in directory: {}".format(
            self.name, self.nevts, self.inputdir))

        self.data = None

    def read(self, evtid: int = 0) -> bool:
        """Read one event from the input directory."""
        filename = self.pyg_files[evtid]
        print("Reading file: {}".format(filename))
        data = torch.load(filename, map_location=torch.device("cpu"))
        self.data = data

        return data

    def get_node_features(self, node_features: List[str],
                          node_scales: Optional[List[float]] = None) -> torch.Tensor:
        """Get the node features from the data object"""
        if self.data is None:
            raise RuntimeError("Please read the data first!")

        node_features = torch.stack([self.data[x] for x in node_features], dim=-1).float()
        if node_scales is not None:
            node_scales = torch.Tensor(node_scales)
            node_features = node_features / node_scales

        return node_features

    def get_edge_masks(self) -> torch.Tensor:
        """Get the masks for edges of interest"""
        if self.data is None:
            raise RuntimeError("Please read the data first!")

        data = self.data
        # edge-level selections
        mask = (data.pt >= 900) & (data.nhits >= 3) & (data.primary == 1) \
            & (data.pdgId != 11) & (data.pdgId != -11)
        return mask
