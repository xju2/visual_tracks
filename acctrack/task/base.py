from acctrack.task.hparams_mixin import HyperparametersMixin

class TaskHooks:

    def run(self) -> None:
        """Run the task"""


class TaskBase(TaskHooks, HyperparametersMixin):
    def __init__(self) -> None:
        super().__init__()