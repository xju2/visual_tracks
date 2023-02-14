contents = """
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
    \"\"\"Comments
    \"\"\"


@hydra.main(config_path=root / "configs", config_name="{script_name}.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    main_function(cfg)


if __name__ == "__main__":
    main()
"""

import os
def create_a_template(script_name):
    os.makedirs("scripts", exist_ok=True)
    post_fix = "py"
    if "." in script_name:
        script_name, post_fix = script_name.split(".")

    with open(f"scripts/{script_name}.{post_fix}", "w") as f:
        f.write(contents.format(script_name=script_name))

    os.makedirs("configs", exist_ok=True)
    with open(f"configs/{script_name}.yaml", "w") as f:
        f.write("")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create a template for a script')
    add_arg = parser.add_argument
    add_arg('script_name', help='script name')
    
    args = parser.parse_args()
    
    create_a_template(args.script_name)