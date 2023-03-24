"""Utilities for reading the ROOT files dumped from the DumpObject."""
from typing import Dict

import numpy as np

def create_arrays(branch_arrays: Dict[str, np.ndarray]):
    variables = list(branch_arrays.keys())
    num_evts = len(branch_arrays[variables[0]])

    all_arrays = []
    for i in range(num_evts):
        inputs = [branch_arrays[x][i] for x in variables]
        array = np.stack(inputs, axis=1)
        all_arrays.append(array)

    return all_arrays


# define particle branch names
# they are taken from
# https://gitlab.cern.ch/gnn4itkteam/athena/-/blob/21.9.26-root-and-csv-files-from-RDO-v1/Tracking/TrkDumpAlgs/src/ROOT2CSVconverter.cpp

particle_branch_prefix = "Part_"
# "vParentID", "vParentBarcode" are not used.
particle_columns = [
    "event_number", "barcode",
    "px", "py", "pz", "pt", "eta",
    "vx", "vy", "vz",
    "radius", "status", "charge", "pdg_id",
    "passed", "vProdNin", "vProdNout", "vProdStatus",
    "vProdBarcode"
]
particle_branch_names = [particle_branch_prefix + col for col in particle_columns]

# cluster branch names
cluster_branch_prefix = "CL"
cluster_columns = [
    "index", "hardware",
    "x", "y", "z",
    "barrel_endcap", "layer_disk", "eta_module", "phi_module", "side",
    "pixel_count", "charge_count",
    "loc_eta", "loc_phi",
    "loc_direction1", "loc_direction2", "loc_direction3",
    "global_eta", "global_phi",
    "eta_angle", "phi_angle", "norm_x", "norm_y", "norm_z",
]
cluster_branch_names = [cluster_branch_prefix + col for col in cluster_columns]

# cluster link to particles
# one cluster may link to multiple particles
coluster_link_columns = [
    "particleLink_eventIndex", "particleLink_barcode"
]
cluster_link_branch_names = [cluster_branch_prefix + col for col in coluster_link_columns]

# spacepoint branch names
spacepoint_branch_prefix = "SP"
spacepoint_columns = [
    "index", "x", "y", "z",
    "CL1_index", "CL2_index",
]
spacepoint_branch_names = [spacepoint_branch_prefix + col for col in spacepoint_columns]
