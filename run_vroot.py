
import logging
from pathlib import Path

import hydra
import pyrootutils
from acctrack import utils
from acctrack.utils import resolvers
from omegaconf import DictConfig

resolvers.add_my_resolvers()


@utils.task_wrapper
def main_function(cfg: DictConfig) -> None:
    """Main function to invoke different tasks"""

    logging.basicConfig(
        filename=Path(cfg.paths.output_dir, "log.txt"),
        encoding="uft-8",
        level=logging.INFO,
    )
    if not cfg.get("task"):
        raise ValueError("Task is not specified in the config file.")

    # Instantiate the task
    task = hydra.utils.instantiate(cfg.task)
    canvas = hydra.utils.instantiate(cfg.canvas)
    task.set_canvas(canvas)

    # histograms
    task.add_histograms(cfg.histograms)

    task.run()


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@hydra.main(
    config_path=str(root / "configs"), config_name="run_vroot_task.yaml", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    main_function(cfg)


if __name__ == "__main__":
    main()
