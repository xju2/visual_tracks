import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import List

import hydra
from omegaconf import DictConfig

from acctrack.task.base import TaskBase
from acctrack import utils

log = utils.get_pylogger(__name__)
@utils.task_wrapper
def main_function(cfg: DictConfig) -> None:
    """Main function to invoke different tasks
    """
    if not cfg.get("task"):
        raise ValueError("Task is not specified in the config file.")
    else:
        print(cfg.task)

    print(cfg.task)
    tasks: List[TaskBase] = utils.instantiate_tasks(cfg.task)
    for task in tasks:
        task.run()

    return True


@hydra.main(config_path=root / "configs", config_name="run_task.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    main_function(cfg)

if __name__ == "__main__":
    main()

