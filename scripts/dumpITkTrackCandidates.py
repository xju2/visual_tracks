#!/usr/bin/env python
"""
The script is to dump the ITkTrackCandidates from the ROOT file produced by DumpObject.
"""

import uproot
import numpy as np
import pandas as pd


def dumpITkTrackCandidates(
    filename: str,
    tree_name: str = "GNN4ITk",
    num_evts: int = 1,
    no_csv: bool = False,
    use_clusters: bool = False,
    use_strip: bool = True,
):
    events = uproot.open(f"{filename}:{tree_name}")
    tracks_info = events.arrays(
        [
            "TRKspacepointsIdxOnTrack",
            "TRKspacepointsIdxOnTrack_trkIndex",
            "TRKspacepointsIsPixel",
            "TRKperigee_position",
            "TRKperigee_momentum",
            "TRKmeasurementsOnTrack_pixcl_sctcl_index",
        ]
    )

    event_info = events.arrays(["event_number", "run_number"])
    spacepoints_info = events.arrays(["SPx", "SPy"])
    num_events = len(event_info["event_number"]) if num_evts == -1 else num_evts
    print(f"Number of events: {num_events}")

    num_tracks = []
    num_clusters_per_track = []
    num_si_clusters_per_track = []
    num_pixel_clusters_per_track = []
    num_sp_per_track = []
    num_non_ghost_sp_per_track = []
    num_strip_sp_per_track = []
    num_non_ghost_strip_sp_per_track = []
    eta_per_track = []
    for ievt in range(num_events):
        event_number = event_info["event_number"][ievt]
        run_number = event_info["run_number"][ievt]
        spacepoint_x = spacepoints_info["SPx"][ievt].to_numpy()
        spacepoint_y = spacepoints_info["SPy"][ievt].to_numpy()

        track_info = tracks_info[ievt]
        hit_id = track_info["TRKspacepointsIdxOnTrack"].to_numpy()
        track_id = track_info["TRKspacepointsIdxOnTrack_trkIndex"].to_numpy()
        is_pixel = track_info["TRKspacepointsIsPixel"].to_numpy()
        perigee_momentum = track_info["TRKperigee_momentum"].to_numpy()
        clusters_on_tracks = track_info["TRKmeasurementsOnTrack_pixcl_sctcl_index"]

        track_info = pd.DataFrame(
            {"hit_id": hit_id, "track_id": track_id, "is_pixel": is_pixel}
        )
        tracks = track_info.groupby("track_id")

        output_str = ""
        itrk = 0
        for track in tracks:
            # calculate the track parameters
            itrk = track[0]
            px, py, pz = perigee_momentum[itrk]
            pT = np.sqrt(px**2 + py**2)
            theta = np.arctan2(pT, pz)
            eta = -np.log(np.tan(theta / 2))
            eta_per_track.append(eta)

            # count the number of clusters in the track candidate
            num_clusters_per_track.append(len(track[1].hit_id))
            num_si_clusters_per_track.append(
                len([x for x in track[1].hit_id if x != -1])
            )
            num_pixel_clusters_per_track.append(
                len(
                    [
                        x
                        for x, y in zip(track[1].hit_id, track[1].is_pixel)
                        if x != 1 and y == 1
                    ]
                )
            )

            # remove duplicated hits while keeping the order
            final_hit_id = []
            strip_sps = []
            pixel_sps = []
            for idx, hit in enumerate(track[1].hit_id):
                if hit == -1:
                    continue

                if hit not in final_hit_id:
                    final_hit_id.append(hit)
                    if track[1].is_pixel.iloc[idx] == 0:  # strip spacepoint
                        strip_sps.append(hit)
                    elif track[1].is_pixel.iloc[idx] == 1:
                        pixel_sps.append(hit)
                    else:
                        pass

            num_sp_per_track.append(len(final_hit_id))
            num_strip_sp_per_track.append(len(strip_sps))

            # remove ghost space points,
            # defined as those have only one cluster in the track candidate
            unique_hits = []
            unique_good_strip_hits = []
            for idx, hit in enumerate(track[1].hit_id[:-1]):
                if hit == -1:
                    continue
                if track[1].is_pixel.iloc[idx] == 0:
                    if hit == track[1].hit_id.iloc[idx + 1]:
                        unique_hits.append(hit)
                        unique_good_strip_hits.append(hit)
                elif track[1].is_pixel.iloc[idx] == 1:
                    unique_hits.append(hit)
                else:
                    pass
            if track[1].hit_id.iloc[-1] != -1 and track[1].is_pixel.iloc[-1] == 1:
                unique_hits.append(track[1].hit_id.iloc[-1])

            num_non_ghost_strip_sp_per_track.append(len(unique_good_strip_hits))
            x_spacepoints_in_track, y_spacepoints_in_track = (
                spacepoint_x[unique_hits],
                spacepoint_y[unique_hits],
            )

            # sort spacepoints by r, r = sqrt(x^2 + y^2)
            r_spacepoints_in_track = np.sqrt(
                x_spacepoints_in_track**2 + y_spacepoints_in_track**2
            )
            sorted_idx = np.argsort(r_spacepoints_in_track)
            unique_hits = np.array(unique_hits)
            unique_hits = unique_hits[sorted_idx]

            num_non_ghost_sp_per_track.append(len(unique_hits))

            if use_clusters:
                output_str += ",".join([str(x) for x in unique_hits[:3]])
                output_str += ","
                cluster_idx = clusters_on_tracks[itrk].to_numpy()
                output_str += ",".join([str(x) for x in cluster_idx])
            else:  # use spacepoints
                if use_strip:
                    output_str += ",".join([str(x) for x in unique_hits])
                else:
                    output_str += ",".join([str(x) for x in pixel_sps])

            output_str += "\n"
            # prepare for the next track
            itrk += 1

        num_tracks.append(itrk)

        if not no_csv:
            with open(
                f"track_{run_number}_{event_number}.csv", "w", encoding="utf-8"
            ) as f:
                f.write(output_str)

    num_tracks = np.array(num_tracks)
    num_clusters_per_track = np.array(num_clusters_per_track)
    num_si_clusters_per_track = np.array(num_si_clusters_per_track)
    num_pixel_clusters_per_track = np.array(num_pixel_clusters_per_track)
    print(f"Mean number of tracks: {num_tracks.mean():.0f}")
    print(f"Mean number of clusters per track: {num_clusters_per_track.mean():.3f}")
    print(
        f"Mean number of silicon clusters per track: {num_si_clusters_per_track.mean():.3f}"
    )
    print(
        f"Mean number of pixel clusters per track: {num_pixel_clusters_per_track.mean():.3f}"
    )

    num_sp_per_track = np.array(num_sp_per_track)
    num_non_ghost_sp_per_track = np.array(num_non_ghost_sp_per_track)
    print(f"Mean number of spacepoints per track: {num_sp_per_track.mean():.3f}")
    print(
        f"Mean number of strip spacepoints per track: {np.array(num_strip_sp_per_track).mean():.3f}"
    )
    print(
        f"Mean number of non-ghost spacepoints per track: {num_non_ghost_sp_per_track.mean():.3f}"
    )
    print(
        f"Mean number of non-ghost strip spacepoints per track: {np.array(num_non_ghost_strip_sp_per_track).mean():.3f}"
    )
    np.savez(
        f"track_info_{tree_name}.npz",
        num_tracks=num_tracks,
        num_clusters_per_track=num_clusters_per_track,
        num_si_clusters_per_track=num_si_clusters_per_track,
        num_pixel_clusters_per_track=num_pixel_clusters_per_track,
        num_sp_per_track=num_sp_per_track,
        num_strip_sp_per_track=num_strip_sp_per_track,
        num_non_ghost_sp_per_track=num_non_ghost_sp_per_track,
        num_non_ghost_strip_sp_per_track=num_non_ghost_strip_sp_per_track,
        eta_per_track=eta_per_track,
    )


def dumpITkTrackDetails(filename: str, tree_name: str = "GNN4ITk"):
    events = uproot.open(f"{filename}:{tree_name}")
    tracks_info = events.arrays(
        [
            "TRKperigee_position",
            "TRKperigee_momentum",
            "TRKmot",
            "TRKoot",
            "TRKcharge",
            "TRKmeasurementsOnTrack_pixcl_sctcl_index",
        ]
    )
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
        measurements_on_track = track_info["TRKmeasurementsOnTrack_pixcl_sctcl_index"]
        mots = track_info["TRKmot"].to_numpy()
        oots = track_info["TRKoot"].to_numpy()
        charges = track_info["TRKcharge"].to_numpy()
        print(perigee_position.shape, perigee_momentum.shape)

        itrk = 1028
        print(f"Track info for {itrk}th track:")
        mot = mots[itrk]
        oot = oots[itrk]
        charge = charges[itrk]
        x, y, z = perigee_position[itrk]
        px, py, pz = perigee_momentum[itrk]
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
        print(f"measurements on track: {mot} and pT = {pT:.3f} [MeV]")
        print(f"outliers on track: {oot}")
        print(f"measurements on track: {measurements_on_track[itrk].to_numpy().shape}")
        break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="The ROOT file produced by DumpObject")
    parser.add_argument("--tree-name", default="GNN4ITk", help="The name of the tree")
    parser.add_argument(
        "--num-evts", type=int, default=1, help="Number of events to process"
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Do not save the track information to csv"
    )
    parser.add_argument(
        "--use-clusters", action="store_true", help="Use clusters to define tracks"
    )
    parser.add_argument(
        "--no-strip", action="store_true", help="Use strip spacepoints to define tracks"
    )
    args = parser.parse_args()

    use_strip = not args.no_strip
    dumpITkTrackCandidates(
        args.filename, args.tree_name, args.num_evts, args.no_csv, args.use_clusters, use_strip
    )
    # dumpITkTrackDetails(args.filename, args.tree_name)
