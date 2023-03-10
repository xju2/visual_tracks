"""Utilities for reading raw athena files dumped from the DumpObject."""
import pandas as pd
import numpy as np

def read_track_candidates(fname):
    """Read track candidates from a file.
    """
    track_candidates = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            track_candidates.append([int(x) for x in line.strip().split(",")])
    return track_candidates


def read_track_candidates_clusters(fname):
    track_candidates = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            track_candidates.append([int(x) for x in line.strip().split(",")])
    return track_candidates


def read_spacepoints(filename):
    spacepoints = pd.read_csv(filename,
                              header=None, engine="python",
                              names=["hit_id", "x", "y", "z", "cluster_index_1", "cluster_index_2"])
    # pixel_hits = spacepoints[pd.isna(spacepoints["cluster_index_2"])]
    # strip_hits = spacepoints[~pd.isna(spacepoints["cluster_index_2"])]
    return spacepoints


def split_spacepoints(spacepoints):
    pixel_hits = spacepoints[pd.isna(spacepoints["cluster_index_2"])]
    strip_hits = spacepoints[~pd.isna(spacepoints["cluster_index_2"])]
    return pixel_hits, strip_hits


def read_particles(filename):        
    field_names = ['subevent', 'barcode', 'px', 'py', 'pz', 'pt', 
            'eta', 'vx', 'vy', 'vz', 'radius', 'status', 'charge', 
            'pdgId', 'pass', 'vProdNIn', 'vProdNOut', 'vProdStatus', 'vProdBarcode']
    particles = pd.read_csv(filename, header=None, engine='python', sep=r",#")
    
    particles = particles[0].str.split(",", expand=True)
    particles.columns = field_names
    particles = particles.astype({"subevent": int, "barcode": int, "px": float, "py": float, "pz": float, "pt": float,
            "eta": float, "vx": float, "vy": float, "vz": float, "radius": float, "status": int, "charge": float,
            "pdgId": int, "pass": str, "vProdNIn": int, "vProdNOut": int, "vProdStatus": int, "vProdBarcode": int})
    
    particle_ids = get_particle_ids(particles)
    particles.insert(0, "particle_id", particle_ids)
    
    return particles


def get_particle_ids(df) -> pd.Series:
    barcode = df.barcode.astype(str)
    subevent = df.subevent.astype(str)

    ## convert bartcode
    # max_length = len(str(barcode.astype(int).max()))
    max_length = 7
    particle_ids = subevent + barcode.str.pad(width=max_length, fillchar='0')
    return particle_ids


def read_true_track(filename):
    """Read fitted tracks information from a file."""
    true_track = pd.read_csv(filename, header=None, engine='python', sep=r",#")

    ## first block on track indices
    trk_index = true_track[0].str.split(",", expand=True)
    trk_index.columns = ['trkid', 'fitter', 'material']
    trk_index = trk_index.astype({'trkid': 'int32', 'fitter': 'int32', 'material': 'int32'})

    ## second block on track info
    trk_info = true_track[3].str.split(",", expand=True).drop(0, axis=1)
    trk_info.columns = [
        'nDoF', "chi2", "charge", "x", 'y', 'z',
        'px', 'py', 'pz', 'mot', 'oot',
        "trkid", 'subevent', 'barcode', 'probability',
        'pdgId', "status"
    ]
    trk_info = trk_info.astype({
        'nDoF': 'int32', "chi2": 'float32', "charge": 'int32', "x": 'float32', 'y': 'float32', 'z': 'float32',
        'px': 'float32', 'py': 'float32', 'pz': 'float32', 'mot': 'int32', 'oot': 'int32',
        "trkid": 'int32', 'subevent': 'int32', 'barcode': 'int32', 'probability': 'float32',
        'pdgId': 'int32', "status": 'int32'
    })
    trk_info = trk_info.drop("trkid", axis=1)

    particle_id = get_particle_ids(trk_info)
    trk_info = trk_info.assign(
        particle_id=particle_id,
        pt=np.sqrt(trk_info.px**2 + trk_info.py**2),
        )

    ## merge the two blocks
    tracks = pd.concat([trk_index, trk_info], axis=1)
    return tracks


def read_detailed_matching(filename):
    detailed_true_tracks = pd.read_csv(filename, header=None, engine='python', sep=r",#")
    num_matched = detailed_true_tracks[0].str.split(",", expand=True)
    num_matched.columns = ['trkid', 'num_matched']
    num_matched = num_matched.astype({'trkid': 'int32', 'num_matched': 'int32'})

    primary_matched = detailed_true_tracks[1].str.split(",", expand=True)[[1, 2]]
    primary_matched.columns = ['subevent', 'barcode']
    primary_matched = primary_matched.astype({'subevent': 'int32', 'barcode': 'int32'})

    true_hits = detailed_true_tracks[2].str.split(",", expand=True)[[1, 2]]
    true_hits.columns = ['true_pixel_hits', 'true_sct_hits']
    true_hits = true_hits.astype({'true_pixel_hits': 'int32', 'true_sct_hits': 'int32'})

    reco_hits = detailed_true_tracks[3].str.split(",", expand=True)[[1, 2]]
    reco_hits.columns = ['reco_pixel_hits', 'reco_sct_hits']
    reco_hits = reco_hits.astype({'reco_pixel_hits': 'int32', 'reco_sct_hits': 'int32'})

    common_hits = detailed_true_tracks[4].str.split(",", expand=True)[[1, 2]]
    common_hits.columns = ['common_pixel_hits', 'common_sct_hits']
    common_hits = common_hits.astype({'common_pixel_hits': 'int32', 'common_sct_hits': 'int32'})

    detailed_matching = pd.concat([num_matched, primary_matched, true_hits, reco_hits, common_hits], axis=1)

    particle_id = get_particle_ids(detailed_matching)
    detailed_matching = detailed_matching.assign(particle_id=particle_id)

    return detailed_matching