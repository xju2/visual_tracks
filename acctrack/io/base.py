"""Base class for IO."""

from acctrack.io import MeasurementData


class BaseReader:
    """Base class for IO."""

    def __init__(self, basedir, name="BaseReader"):
        """Initialize
        basedir: input directory
        name: name of the reader
        """
        self.name = name
        self.basedir = basedir
        self.all_evtids = []
        self.nevts = 0

    def read(self, evtid: int = None) -> MeasurementData:
        """Read one event from the input directory."""
        raise NotImplementedError

    def  __call__(self, evtid: int = None):
        return self.read(evtid)

    def __str__(self):
        return "{} reads from {}.".format(self.name, self.basedir)