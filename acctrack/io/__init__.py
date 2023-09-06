from acctrack.io.base import BaseTrackDataReader
from acctrack.io.acts import ActsReader
from acctrack.io.athena_data import AthenaDFReader
from acctrack.io.trackml import TrackMLReader

__all__ = [
    "BaseTrackDataReader" "ActsReader",
    "AthenaDFReader",
    "AthenaRawDataReader",
    "TrackMLReader",
]

try:
    import uproot
    from acctrack.io.athena_raw_data import AthenaRawDataReader

    __all__.append("AthenaRawDataReader")
except ImportError:
    print("uproot is not installed. AthenaRawRootReader will not be available.")
