from collections import namedtuple

MeasurementData = namedtuple('MeasurementData', [
    'hits', 'measurements', 'meas2hits', 'spacepoints',
    'particles', 'true_edges', 'event_file'])