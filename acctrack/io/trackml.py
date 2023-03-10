"""Read TrackML data.
https://www.kaggle.com/c/trackml-particle-identification/overview

Assume the files are like:
detectors.csv
trackml/event000001000-hits.csv
trackml/event000001000-cells.csv
trackml/event000001000-particles.csv
trackml/event000001000-truth.csv
"""
import os
import glob
import re

import numpy as np
import pandas as pd

from acctrack.io.base import BaseMeasurementDataReader
from acctrack.io import MeasurementData
from acctrack.io.utils_feature_store import make_true_edges
from acctrack.io.trackml_cell_info import add_cluster_shape
from acctrack.io.trackml_detector import load_detector

__all__ = ['TrackMLReader', 'select_barrel_hits', 'remove_noise_hits']

def select_barrel_hits(hits):
    """Select barrel hits.
    """
    vlids = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    return hits

def remove_noise_hits(hits):
    """Remove noise hits.
    """
    # Remove noise hits
    hits = hits[hits.hit_type != 'noise']
    return hits

class TrackMLReader(BaseMeasurementDataReader):
    def __init__(self, basedir, name="TrackMLReader") -> None:
        super().__init__(basedir, name)

        # count how many events in the directory
        all_evts = glob.glob(os.path.join(
            self.basedir, "event*-hits.csv"))

        self.nevts = len(all_evts)
        pattern = "event([0-9]*)-hits.csv"
        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip())
                for x in all_evts])

        print("total {} events in directory: {}".format(
            self.nevts, self.basedir))
        
        detector_path = os.path.join(self.basedir, "../detectors.csv")
        _, self.detector = load_detector(detector_path)


    def read(self, evtid: int = None) -> MeasurementData:
        """Read one event from the input directory"""

        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print("read event {}".format(evtid))

        prefix = os.path.join(self.basedir, "event{:09d}".format(evtid))
        hit_fname = "{}-hits.csv".format(prefix)
        cell_fname = "{}-cells.csv".format(prefix)
        particle_fname = "{}-particles.csv".format(prefix)
        truth_fname = "{}-truth.csv".format(prefix)

        # read all files
        hits = pd.read_csv(hit_fname)
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        hits = hits.assign(r=r, phi=phi)

        # read truth info about hits and particles
        truth = pd.read_csv(truth_fname)
        particles = pd.read_csv(particle_fname)
        ### add dummy particle information for noise hits
        ### whose particle ID is zero.
        ### particle_id,vx,vy,vz,px,py,pz,q,nhits
        particles.loc[len(particles.index)] = [0, 0, 0, 0, 0.00001, 0.00001, 0.00001, 0, 0]
        truth.merge(particles, on='particle_id', how='left')
        truth = truth.assign(pt=np.sqrt(truth.px**2 + truth.py**2))


        hits = hits.merge(truth[['hit_id', 'particle_id',
            'vx', 'vy', 'vz', 'pt', 'weight']], on='hit_id')

        true_edges = make_true_edges(hits)
        cells = pd.read_csv(cell_fname)
        hits = add_cluster_shape(hits, cells, self.detector)

        hits = hits.assign(R=np.sqrt( (hits.x - hits.vx)**2 + (hits.y - hits.vy)**2 + (hits.z - hits.vz)**2 ))
        hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)

        data = MeasurementData(
            hits=None,
            measurements=None,
            meas2hits=None,
            spacepoints=hits,
            particles=particles,
            true_edges=true_edges,
            event_file=os.path.abspath(prefix),
        )
        return data

