"""Utilities for feature stores
* read feature files that used for embedding.
* save measurement data for training embedding.
"""
from gnn4itk.io import MeasurementData

import numpy as np

scales = np.array([1000, np.pi, 1000], dtype=np.float32)

def read_torch(filename, edge_name='layerless_true_edges'):
    import torch
    data = torch.load(filename, map_location='cpu')

    hits = data['x'].detach().numpy()
    pids = data['pid'].detach().numpy()
    true_edges = data[edge_name].detach().numpy()
    return hits, pids, true_edges


def read_np(fname, edge_name='layerless_true_edges'):
    import numpy as np
    arrays = np.load(fname)
    hits, pids, true_edges = arrays['x'], arrays['pid'], arrays[edge_name]
    return hits, pids, true_edges


def read(filename, edge_name='layerless_true_edges', is_torch_data=False):
    read_fn = read_torch if is_torch_data else read_np
    return read_fn(filename, edge_name)


particle_features = [
    'particle_id', 'pt', 'eta',
    'charge', 'vx', 'vy', 'vz', 'radius'
]
def save_to_np(outname, data: MeasurementData):
    hits = data.spacepoints
    edges = data.true_edges
    np.savez_compressed(
        outname, x=hits[['r', 'phi', 'z']].values/scales,
        pid=hits['particle_id'].values,
        hid=hits['hit_id'].values,
        layerless_true_edges=edges,
        particles=data.particles[particle_features].values
    )

def save_to_torch(outname, measurement_data):
    from torch_geometric.data import Data
    import torch
    sp_hits = measurement_data.spacepoints
    edges = measurement_data.true_edges
    prefix = measurement_data.event_file
    data = Data(
        x=torch.from_numpy(sp_hits[['r', 'phi', 'z']].values).float()/scales,
        pid=torch.from_numpy(sp_hits['particle_id'].values),
        # layers=torch.from_numpy(sp_hits['geometry_id'].values),
        event_file=prefix,
        hid=torch.from_numpy(sp_hits['hit_id'].values),
        layerless_true_edges=torch.from_numpy(edges),
        particles=torch.from_numpy(data.particles[particle_features].values),
    )
    with open(outname, 'wb') as f:
        torch.save(data, f)