from collections import namedtuple

__all__ = ['TrackMLReader', 'ActsReader', "MeasurementData"]

from acctrack.io.trackml import TrackMLReader
from acctrack.io.acts import ActsReader

MeasurementData = namedtuple('MeasurementData', [
    'hits', 'measurements', 'meas2hits', 'spacepoints',
    'particles', 'true_edges', 'event_file'])