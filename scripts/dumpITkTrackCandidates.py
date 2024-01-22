#!/usr/bin/env python
"""
The script is to dump the ITkTrackCandidates from the ROOT file produced by DumpObject.
"""

import uproot
import numpy as np
import pandas as pd

def dumpITkTrackCandidates(filename: str, tree_name: str = "GNN4ITk"):
    events = uproot.open(f"{filename}:{tree_name}")
    tracks_info = events.arrays(["TRKspacepointsIdxOnTrack", "TRKspacepointsIdxOnTrack_trkIndex"])
    event_info = events.arrays(["event_number", "run_number"])
    num_events = len(event_info["event_number"])
    print(f"Number of events: {num_events}")

    num_tracks = []
    num_hits_per_track = []
    for ievt in range(num_events):
        event_number = event_info["event_number"][ievt]
        run_number = event_info["run_number"][ievt]

        track_info = tracks_info[ievt]
        hit_id = track_info["TRKspacepointsIdxOnTrack"].to_numpy()
        track_id = track_info["TRKspacepointsIdxOnTrack_trkIndex"].to_numpy()

        track_info = pd.DataFrame({"hit_id": hit_id, "track_id": track_id})
        tracks = track_info.groupby("track_id")

        output_str = ""
        itrk = 0
        for track in tracks:
            if len(tracks) < 4:
                continue

            silicon_hits = [x for x in track[1].hit_id if x != -1]
            # remove duplicated hits while keeping the order
            final_hit_id = []
            for hit in silicon_hits:
                if hit not in final_hit_id:
                    final_hit_id.append(hit)

            if len(final_hit_id) < 4:
                continue

            itrk += 1
            output_str += ",".join([str(x) for x in final_hit_id])
            output_str += "\n"
            num_hits_per_track.append(len(final_hit_id))
            print(track)
            print(silicon_hits)
            print(final_hit_id)
        num_tracks.append(itrk)

        with open(f"track_{run_number}_{event_number}.csv", "w", encoding="utf-8") as f:
            f.write(output_str)
        break

    num_tracks = np.array(num_tracks)
    num_hits_per_track = np.array(num_hits_per_track)
    print(f"Mean number of tracks: {num_tracks.mean():.0f}")
    print(f"Mean number of hits per track: {num_hits_per_track.mean():.2f}")
    np.savez(f"track_info_{tree_name}.npz",
             num_tracks=num_tracks,
             num_hits_per_track=num_hits_per_track)

def dumpITkTrackDetails(filename: str, tree_name: str = "GNN4ITk"):
    events = uproot.open(f"{filename}:{tree_name}")
    tracks_info = events.arrays(["TRKperigee_position", "TRKperigee_momentum", "TRKmot", "TRKcharge"])
    event_info = events.arrays(["event_number", "run_number"])
    num_events = len(event_info["event_number"])
    print(f"Number of events: {num_events}")

    for ievt in range(num_events):
        event_number = event_info["event_number"][ievt]
        run_number = event_info["run_number"][ievt]
        print(f"checking event: {run_number} {event_number}")

        track_info = tracks_info[ievt]
        perigee_position = track_info["TRKperigee_position"].to_numpy()
        perigee_momentum = track_info["TRKperigee_momentum"].to_numpy()
        mots = track_info["TRKmot"].to_numpy()
        charges = track_info["TRKcharge"].to_numpy()
        print(perigee_position.shape, perigee_momentum.shape)
        print("first track info:")
        mot = mots[0]
        charge = charges[0]
        x, y, z = perigee_position[0]
        px, py, pz = perigee_momentum[0]
        pT = np.sqrt(px**2 + py**2)
        momentum = np.sqrt(px**2 + py**2 + pz**2)
        theta = np.arctan2(pT, pz)
        phi = np.arctan2(py, px)
        # pos_phi = np.arctan2(y, x)
        # pos_theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        # d0 = np.sqrt(x**2 + y**2)
        # eta = -np.log(np.tan(theta / 2))
        print(f"position: ({x:.3f}, {y:.3f}, {z:.3f}) [mm]")
        print(f"momentum: ({px:.3f}, {py:.3f}, {pz:.3f}) [MeV]")
        print(f"phi: {phi:.3f}")
        print(f"theta: {theta:.3f}")
        print(f"qoverp: {charge / momentum:.6f} [1/MeV]")
        print(f"measurements on track: {mot}")
        break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="The ROOT file produced by DumpObject")
    parser.add_argument("--tree-name", default="GNN4ITk", help="The name of the tree")
    args = parser.parse_args()

    dumpITkTrackCandidates(args.filename, args.tree_name)
    # dumpITkTrackDetails(args.filename, args.tree_name)
