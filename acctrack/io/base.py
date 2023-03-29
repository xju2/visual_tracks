from typing import List, Optional
from pathlib import Path
import pandas as pd

class BaseTrackDataReader:
    """Base class for reading Tracking data"""

    def __init__(self, inputdir: str, output_dir: str = None,
                 overwrite: bool = True, name="BaseTrackDataReader"):
        self.inputdir = Path(inputdir)
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(f"Input directory {self.inputdir} does not exist or is not a directory.")

        self.outdir = Path(output_dir) if output_dir else self.inputdir / "processed_data"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.overwrite = overwrite

        # file systems and basic event information
        self.all_evtids: List[int] = []
        self.nevts = 0

        # following are essential dataframe
        self.particles: pd.DataFrame = None
        self.clusters: pd.DataFrame = None
        self.spacepoints: pd.DataFrame = None
        # truth is the same as spacepoints, but contains truth information
        self.truth: pd.DataFrame = None
        self.true_edges: pd.DataFrame = None

        # following are optional dataframe
        # they are created from the dumping object from Athena
        # needed for truth studies
        self.tracks_clusters: Optional[pd.DataFrame] = None
        self.tracks: Optional[pd.DataFrame] = None
        self.true_tracks: Optional[pd.DataFrame] = None
        self.detailed_matching: Optional[pd.DataFrame] = None
        self.tracks_matched_to_truth: Optional[pd.DataFrame] = None

    def read(self, evtid: int = 0) -> bool:
        """Read one event from the input directory."""
        raise NotImplementedError

    def __str__(self):
        return "{} reads from {}.".format(self.name, self.inputdir)
