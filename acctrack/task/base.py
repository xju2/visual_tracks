from acctrack.task.hparams_mixin import HyperparametersMixin

class TaskHooks:

    def run(self) -> bool:
        """Run the task"""


class TaskBase(TaskHooks, HyperparametersMixin):
    def __init__(self, name="TaskBase") -> None:
        super().__init__()
        self.name = name