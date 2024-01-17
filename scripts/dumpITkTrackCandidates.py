#!/usr/bin/env python
"""
The script is to dump the ITkTrackCandidates from the ROOT file produced by DumpObject.
"""

import uproot
import pandas as pd

def dumpITkTrackCandidates(filename: str, tree_name: str = "GNN4ITk"):
    events = uproot.open(f"{filename}:{tree_name}")
    track_info = events.arrays(["TRKspacepointsIdxOnTrack", "TRKspacepointsIdxOnTrack_trkIndex"])
    event_info = events.arrays(["event_number", "run_number"])
    num_events = len(event_info["event_number"])

    output_str = ""
    for ievt in range(num_events):
        event_number = event_info["event_number"][ievt]
        run_number = event_info["run_number"][ievt]

        tracks_info = track_info[ievt]
        hit_id = tracks_info["TRKspacepointsIdxOnTrack"].to_numpy()
        track_id = tracks_info["TRKspacepointsIdxOnTrack_trkIndex"].to_numpy()

        tracks_info = pd.DataFrame({"hit_id": hit_id, "track_id": track_id})
        tracks = tracks_info.groupby("track_id")
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

            output_str += ",".join([str(x) for x in final_hit_id])
            output_str += "\n"

        with open(f"track_{run_number}_{event_number}.csv", "w", encoding="utf-8") as f:
            f.write(output_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="The ROOT file produced by DumpObject")
    parser.add_argument("--tree-name", default="GNN4ITk", help="The name of the tree")
    args = parser.parse_args()

    dumpITkTrackCandidates(args.filename, args.tree_name)
