"""Read csv files obtained from ACTS and return """
import os
import re
import glob
from typing import Any

import numpy as np
import pandas as pd
import itertools

from gnn4itk.io import MeasurementData

def true_edges(hits):
    hit_list = hits.groupby(['particle_id', 'geometry_id'],
        sort=False)['index'].agg(lambda x: list(x)).groupby(
            level=0).agg(lambda x: list(x))

    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    layerless_true_edges = np.array(e).T
    return layerless_true_edges


class ACTSCSVReader:
    def __init__(self, basedir, spname='spacepoint', *args, **kwargs):
        self.basedir = basedir
        self.spname = spname

        # count how many events in the directory
        all_evts = glob.glob(os.path.join(
            self.basedir, "event*-{}.csv".format(spname)))

        self.nevts = len(all_evts)
        pattern = "event([0-9]*)-{}.csv".format(spname)
        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip())
                for x in all_evts])
        print("total {} events in directory: {}".format(
            self.nevts, self.basedir))


    def read(self, evtid: int = None):
        """Read one event from the input directory.
        
        Return:
            MeasurementData
        """
        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
        
        prefix = os.path.join(self.basedir, "event{:09d}".format(evtid))
        hit_fname = "{}-hits.csv".format(prefix)
        measurements_fname = "{}-measurements.csv".format(prefix)
        measurements2hits_fname = "{}-measurement-simhit-map.csv".format(prefix)
        sp_fname = '{}-{}.csv'.format(prefix, self.spname)
        p_name = '{}-particles_final.csv'.format(prefix)

        # read hit files
        hits = pd.read_csv(hit_fname)
        hits = hits[hits.columns[:-1]]
        hits = hits.reset_index().rename(columns = {'index':'hit_id'})

        # read measurements
        measurements = pd.read_csv(measurements_fname)
        meas2hits = pd.read_csv(measurements2hits_fname)
        sp = pd.read_csv(sp_fname)

        # read particles and add more variables
        particles = pd.read_csv(p_name)
        pt = np.sqrt(particles.px**2 + particles.py**2)
        momentum = np.sqrt(pt**2 + particles.pz**2)
        theta = np.arccos(particles.pz/momentum)
        eta = -np.log(np.tan(0.5*theta))
        radius = np.sqrt(particles.vx**2 + particles.vy**2)
        particles = particles.assign(pt=pt, radius=radius, eta=eta)

        sp_hits = sp.merge(meas2hits, on='measurement_id', how='left').merge(
            hits, on='hit_id', how='left')
        sp_hits = sp_hits.merge(particles, on='particle_id', how='left')

        r = np.sqrt(sp_hits.x**2 + sp_hits.y**2)
        phi = np.arctan2(sp_hits.y, sp_hits.x)
        sp_hits = sp_hits.assign(r=r, phi=phi)

        sp_hits = sp_hits.assign(R=np.sqrt(
            (sp_hits.x - sp_hits.vx)**2
            + (sp_hits.y - sp_hits.vy)**2 
            + (sp_hits.z - sp_hits.vz)**2))
        sp_hits = sp_hits.sort_values('R').reset_index(
            drop=True).reset_index(drop=False)

        edges = true_edges(sp_hits)
        
        data = MeasurementData(
            hits, measurements, meas2hits,
            sp_hits, particles, edges, os.path.abspath(prefix))
        return data

    def __call__(self, evtid: int = None, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)