import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig

from acctrack import utils

log = utils.get_pylogger(__name__)
@utils.task_wrapper
def main_function(cfg: DictConfig) -> None:
    """Main function to invoke different tasks
    """
    pass


@hydra.main(config_path=root / "configs", config_name="run_task.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    main_function(cfg)

if __name__ == "__main__":
    main()

