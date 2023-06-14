"""Convert GNN tracks to csv files for each event for track fitting."""
from pathlib import Path
import pandas as pd
import numpy as np

from acctrack.task.base import TaskBase
from acctrack.utils import get_pylogger

logger = get_pylogger(__name__)

class ConvertGNNTracksForFitting(TaskBase):

    def __init__(self,
                 evtid_matching_fname: str,
                 rdo_matching_fname: str,
                 gnn_track_fname: str,
                 process_file_path: str,
                 output_dir: str,
                 max_evts: int = 512,
                 num_workers: int = 1,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def read_evt_info(self):
        self.evt_info = pd.read_csv(
            self.hparams.evtid_matching_fname, sep='\t', header=None,
            names=["evtID", "orgEvtID", "rdoEvtID", "rdoNum"])

        logger.info("rdo event IDs")
        logger.info(", ".join([str(x) for x in self.evt_info.rdoEvtID.values.tolist()]))

        self.evtid_map = dict(zip(
            self.evt_info.evtID.values.tolist(),
            self.evt_info.rdoEvtID.values.tolist()))


    def read_rdo_info(self):
        self.rdo_info = pd.read_csv(
            self.hparams.rdo_matching_fname, sep='\t', header=None,
            names=["rdoFileName", "rodNum"])


    def read_gnn_tracks(self):
        self.recoTracks = [dict() for _ in range(self.hparams.max_evts)]

        methods = ["singleCutFilter", "wrangler"]
        file_reco = self.hparams.gnn_track_fname
        with pd.HDFStore(file_reco, mode='r') as reader:
            for eventId in range(self.hparams.max_evts):
                for m in methods:
                    dataname = "/event{0}/{1}/reco_tracks".format(eventId, m)
                    if dataname not in reader:
                        continue
                    df_trks = reader.get(dataname)
                    # Remove -1 that are placeholders for empty hit
                    trks = df_trks.values
                    trks = [list(filter(lambda x: x != -1, trk)) for trk in trks]

                    self.recoTracks[eventId].update({m : trks})

    def write_one_evt(self, evtid: int):
        """Write track candidates to a file for a given event ID"""
        try:
            real_evtid = self.evtid_map[evtid]
        except KeyError:
            logger.error("event id {} not there".format(evtid))
            return

        min_num_hits = 3

        outdir = Path(self.hparams.output_dir)
        outname = f"tracks_{real_evtid}.txt"
        if not outdir.exists():
            outdir.mkdir(parents=True)

        outname = outdir / outname

        # read processed data info
        # use that to sort the track candidates
        processed_data_dir = Path(self.hparams.process_file_path)
        truth = None
        if processed_data_dir.is_dir():
            truth_fname = processed_data_dir / f"event{evtid:09}-truth.csv"
            if not truth_fname.exists():
                print(f"{evtid} does not have processed data, no sortting")
            else:
                truth = pd.read_csv(truth_fname)
        else:
            pass

        methods = ['singleCutFilter', 'wrangler']
        with open(outname, 'w') as f:
            for method in methods:
                if method not in self.recoTracks[evtid]:
                    print(f"no reco track for method {method} for event {evtid}")
                    continue

                for track in self.recoTracks[evtid][method]:
                    if len(track) < min_num_hits:
                        continue

                    if truth is None:
                        f.write(','.join([str(i) for i in track]))
                        f.write("\n")
                    else:
                        track_info = truth[truth['hit_id'].isin(track)]
                        # remove duplicated spacepoints
                        track_info = track_info.drop_duplicates(subset='hit_id')
                        # sort by r direction
                        track_info['r'] = np.sqrt(track_info.x**2 + track_info.y**2)
                        track_info = track_info.sort_values(by='r')

                        f.write(','.join([str(i) for i in track_info.hit_id.values.tolist()]))
                        f.write("\n")


    def run(self) -> None:
        self.read_evt_info()
        self.read_rdo_info()
        self.read_gnn_tracks()

        num_workers = self.hparams.num_workers
        evt_ids = self.evt_info.evtID.values.tolist()
        if num_workers > 1:
            from multiprocessing import Pool
            with Pool(num_workers) as pool:
                pool.map(self.write_one_evt, evt_ids)
        else:
            for evtid in evt_ids:
                self.write_one_evt(evtid)
