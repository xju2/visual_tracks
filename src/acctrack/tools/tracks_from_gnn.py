#!/usr/bin/env python
"""
Get track candidates using GNN score.
"""

import sys
import os

import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
import pandas as pd

import torch
import networkx as nx


def build_tracks_dbscan(
    hit_id,
    score,
    senders,
    receivers,
    edge_score_cut=0.0,
    epsilon=0.25,
    min_samples=2,
    **kwargs
):
    n_nodes = hit_id.shape[0]
    if edge_score_cut > 0:
        cuts = score > edge_score_cut
        score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]

    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.sparse.csr_matrix(
        (score, (senders, receivers)), shape=(n_nodes, n_nodes), dtype=np.float32
    )
    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1] / 2.0
    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    # make it symmetric
    e_csr_bi = sp.sparse.coo_matrix(
        (
            np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]),
            np.hstack(
                [
                    np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),
                    np.vstack([e_csr.tocoo().col, e_csr.tocoo().row]),
                ]
            ),
        )
    )

    # DBSCAN get track candidates
    clustering = DBSCAN(
        eps=epsilon, metric="precomputed", min_samples=min_samples
    ).fit_predict(e_csr_bi)
    track_labels = np.vstack(
        [np.unique(e_csr_bi.tocoo().row), clustering[np.unique(e_csr_bi.tocoo().row)]]
    )
    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]
    new_hit_id = np.apply_along_axis(lambda x: hit_id[x], 0, track_labels.hit_id.values)
    tracks = pd.DataFrame.from_dict(
        {"track_id": track_labels.track_id, "hit_id": new_hit_id}
    )
    return tracks



def cc_and_walk(
    hit_id,
    score,
    senders,
    receivers
):
    G = nx.Graph()
    num_nodes = hit_id.shape[0]

    G.add_nodes_from(np.arange(num_nodes), hit_id=hit_id)
    G.add_edges_from(zip(senders, receivers))
    G.node_attr_dict_factory

def process(filename, torch_data_dir, outdir, method_name, score_name, **kwargs):
    evtid = int(os.path.basename(filename)[:-4])
    array = np.load(filename)
    score = array[score_name]
    senders = array["senders"]
    receivers = array["receivers"]

    # torch_fname = os.path.join(torch_data_dir, "{:04}".format(evtid))
    torch_fname = os.path.join(torch_data_dir, "event{:09}.pyg".format(evtid))
    data = torch.load(torch_fname, map_location="cpu")
    hit_id = data["hit_id"].numpy()

    build_track_fn = getattr(sys.modules[__name__], method_name)
    predicted_tracks = build_tracks_dbscan(hit_id, score, senders, receivers, **kwargs)

    # save reconstructed tracks into a file
    np.savez(
        os.path.join(outdir, "{}.npz".format(evtid)),
        predicts=predicted_tracks,
    )
