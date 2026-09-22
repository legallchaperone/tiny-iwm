"""Compose a configuration and write local run records without starting training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir

from utils.provenance import write_run_records


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_DIR = PROJECT_ROOT / "configurations"


def compose_and_write(output_dir: Path, overrides: Sequence[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    return write_run_records(cfg, output_dir, PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    compose_and_write(args.output_dir, overrides)


if __name__ == "__main__":
    main()
