import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import pyrallis
import torch
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src import (
    ExperimentType,
    get_barycenter_and_losses,
    get_experiment,
    log_results,
    setup_logger,
)

# if allowed, can lead to negative GW loss values!
torch.backends.cuda.matmul.allow_tf32 = False


class InitType(str, Enum):
    product = "product"
    random_product = "random_product"
    random = "random"
    fluid = "fluid"
    fluid_soft = "fluid_soft"
    spectral = "spectral"
    spectral_soft = "spectral_soft"
    kmeans = "kmeans"
    kmeans_soft = "kmeans_soft"
    kmeans_torch = "kmeans_torch"
    kmeans_torch_soft = "kmeans_torch_soft"
    spectral_torch = "spectral_torch"
    spectral_torch_soft = "spectral_torch_soft"


INIT_RANDOM = [
    InitType.product,
    InitType.random_product,
    InitType.random,
]

# Register a custom encoder for dumping
pyrallis.encode.register(ExperimentType, lambda x: x.value)
pyrallis.encode.register(InitType, lambda x: x.value)


@dataclass
class NodeRangeConfig:
    n_start: int = 100
    n_end: int = 5000
    n_step: int = 100


@dataclass
class PathConfig:
    scratch_path: Path = Path("/scratch/cdufour/")
    res_dir: Optional[str] = None
    experiment_name: Optional[str] = None
    # Private field to store the resolved full path
    path_exp: Optional[Path] = field(default=None)

    def resolve(
        self,
        experiment_id: int,
        experiment_type: ExperimentType,
        initialization: InitType,
        warmstartT: bool,
        number_graphs: int = 1,
    ) -> Path:
        """
        Replaces the directory creation logic from 'process_args'.  Computes
        experiment_name and path_exp if they are missing.
        """
        # Default experiment name if None: use the ID formatted as 2 digits
        if self.experiment_name is None:
            self.experiment_name = f"{experiment_id:02d}"

        if self.res_dir is None:
            ending = "_warmstartT" if warmstartT else "_coldstartT"
            if number_graphs > 1:
                ending += f"_nGraphs{number_graphs}"
            self.res_dir = f"{experiment_type.value}/{initialization.value}"
            self.res_dir += ending + "/"

        # Logic to construct the full path if not explicitly provided
        if self.path_exp is None:
            self.path_exp = self.scratch_path / self.res_dir / self.experiment_name

        # Create the directory
        os.makedirs(self.path_exp, exist_ok=True)
        return self.path_exp

    @property
    def full_path(self) -> Path:
        if self.path_exp is None:
            raise ValueError("Path not resolved. Call config.setup() first.")
        return self.path_exp


@dataclass
class LogConfig:
    log_filename: Optional[str] = None
    log_level_file: int = 10  # (9) VALS for debugging
    log_level_cli: int = 20  # (20) INFO

    def setup(self, log_dir: Path):
        """Sets up the logger configuration."""
        # Remove default handlers to avoid duplicates
        logger.remove()
        setup_logger(
            path_to_log_folder=log_dir,
            file_name=self.log_filename,
            log_level_file=self.log_level_file,
            log_level_cli=self.log_level_cli,
        )


@dataclass
class ExperimentConfig:
    experiment_id: int
    experiment_type: ExperimentType = ExperimentType.continuous
    number_graphs: int = 1
    num_reps: int = 20
    num_start: int = 1
    initialization: InitType = InitType.product
    seed: int = 1233545625
    rep_seed_offset: int = 0
    warmstartT: bool = True
    timing: bool = False

    nodes: NodeRangeConfig = field(default_factory=NodeRangeConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    def validate(self):
        if self.initialization not in INIT_RANDOM and self.num_start > 1:
            logger.warning(f"WARNING: Using '{self.initialization}' with multiple random starts. Forcing num_start=1.")
            self.num_start = 1

    def setup(self):
        # 1. Resolve Paths (creates directories)
        self.paths.resolve(
            self.experiment_id,
            self.experiment_type,
            self.initialization,
            self.warmstartT,
            self.number_graphs,
        )

        # 2. Setup Logger
        self.logging.setup(self.paths.full_path)

    def save(self):
        """Saves the configuration to a JSON file in the experiment path."""
        config_path = self.paths.full_path / "config.yaml"
        pyrallis.dump(self, open(config_path, "w"))


@torch.inference_mode()
def main():
    cfg = pyrallis.parse(config_class=ExperimentConfig)
    cfg.setup()
    cfg.validate()
    cfg.save()
    logger.info(f"Experiment initialized:  \n{pyrallis.dump(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    since = time()
    experiment = get_experiment(
        id=cfg.experiment_id,
        experiment_type=cfg.experiment_type,
        number_graphs=cfg.number_graphs,
        device=device,
    )

    logger.info(f"Starting experiment {cfg.paths.experiment_name}")

    for n in range(cfg.nodes.n_start, cfg.nodes.n_end + 1, cfg.nodes.n_step):
        k = experiment.get_k(n)
        logger.info(f"Starting n={n:<6}, k<={k}")

        for rep in range(cfg.num_reps):
            # set random seed for internal rep -> can add more reps if needed with another
            # run by modifying cfg.rep_seed_offset
            torch.manual_seed(cfg.seed + rep + cfg.rep_seed_offset)
            np.random.seed(cfg.seed + rep + cfg.rep_seed_offset)

            logger.debug(f"Repetition {rep:<3} for n={n}")

            exp_infos = {
                "n": n,
                "rep": rep,
                "experiment_id": cfg.experiment_id,
                "k_max": k,
            }

            As, ps, thetas, lambdas = experiment.get_data(n=n)

            results = get_barycenter_and_losses(
                k=k,
                As=As,
                thetas=thetas,
                ps=ps,
                lambdas=lambdas,
                initialization=cfg.initialization,
                multistart=cfg.num_start,
                warmstartT=cfg.warmstartT,
                timing=cfg.timing,
            )

            if cfg.timing:
                logger.debug(f"Time for n={n}, rep={rep}: {results['time']:.4f} seconds")

            results |= exp_infos

            # Accessing logging config
            if cfg.logging.log_level_cli <= 9 or cfg.logging.log_level_file <= 9:
                log_results(results)

            # Using Pathlib for saving (cleaner than os.path.join)
            save_path = cfg.paths.full_path / f"res_n{n}_rep{rep}.pt"
            torch.save(results, save_path)

            logger.debug(f"Results for n={n}, rep={rep} saved.")
            if device.type == "cuda":
                for A in As:
                    del A
                for theta in thetas:
                    del theta
                del results
                torch.cuda.empty_cache()
                # check_gpu_memory()
                logger.debug("Garbage collection done.")

    logger.info(f"Saved in {cfg.paths.full_path}")
    logger.info(f"All experiments completed in {timedelta(seconds=time() - since)}")


if __name__ == "__main__":
    main()
