import os
from pathlib import Path

from acctrack.task.base import TaskBase

from acctrack.io.base import BaseReader
from acctrack.io.utils import save_to_np, load_from_np, dump_data
from acctrack.viewer import viewer

class ProcessRawTrackData(TaskBase):
    def __init__(self, reader: BaseReader, out_dir: str, 
                 check: bool = False, 
                 num_workers: int = 1, 
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["reader"])

        self.reader = reader

    def run_one_evt(self, evtid):
        output_dir = self.hparams.out_dir
        check_only = self.hparams.check
        filename = os.path.join(output_dir, str(evtid))
        if check_only:
            data = load_from_np( filename + ".npz")
            dump_data(data)
            viewer.view_graph(data[0], data[1], data[2], outname=filename+"_graph.png")
        else:
            data = self.reader(evtid)
            save_to_np(filename, data)

    def run(self) -> None:
        check = self.hparams.check
        if check:
            evtid = self.reader.all_evtids[0]
            self.run_one_evt(evtid)
        else:
            num_workers = self.hparams.num_workers
            if num_workers < 2:
                for evtid in self.reader.all_evtids:
                    self.run_one_evt(evtid)
            else:
                from multiprocessing import Pool
                with Pool(num_workers) as p:
                    p.map(self.run_one_evt, self.reader.all_evtids)
