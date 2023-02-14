
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

## add your comments
import hydra
from omegaconf import DictConfig

def main_function(cfg: DictConfig) -> None:
    """Comments
    """


@hydra.main(config_path=root / "configs", config_name="convert_gnn_tracks.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    main_function(cfg)


if __name__ == "__main__":
    main()
