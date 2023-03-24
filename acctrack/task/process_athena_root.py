"""Process the raw athena root data and save them to parquet files for training NNs."""
from acctrack.task.base import TaskBase

from acctrack.io.athena_raw_root import AthenaRawRootReader

class ProcessRawAthenaRoot(TaskBase):
    def __init__(self, reader: AthenaRawRootReader,
                 num_workers: int = 1,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["reader"])

        self.reader = reader

    def run_one_task(self, task_idx):
        self.reader.read_file(task_idx)

    def run(self) -> None:
        num_workers = self.hparams.num_workers
        if num_workers < 2:
            for idx in range(self.reader.num_files):
                self.run_one_task(idx)
        else:
            from multiprocessing import Pool
            with Pool(num_workers) as p:
                p.map(self.run_one_task, list(range(self.reader.num_files)))
