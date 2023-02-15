from acctrack.task.base import TaskBase
from acctrack.utils import get_pylogger

logger = get_pylogger(__name__)

class TestTask(TaskBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

    def run(self) -> None:
        logger.info("This is the Test Task.")
        logger.info(self.hparams)