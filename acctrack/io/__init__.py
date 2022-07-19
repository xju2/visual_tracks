from collections import namedtuple
MeasurementData = namedtuple('MeasurementData', [
    'hits', 'measurements', 'meas2hits', 'spacepoints',
    'particles', 'true_edges', 'event_file'])



from acctrack.io.trackml import TrackMLReader
from acctrack.io.acts import ActsReader

__all__ = ['TrackMLReader', 'ActsReader', "MeasurementData"]