import gc
import json
import os
import sys

import torch
from loguru import logger


def setup_logger(
    path_to_log_folder, file_name=None, log_level_file=9, log_level_cli=10
):
    logger.remove()
    logger.level("vals", no=9, color="<yellow>")

    def custom_formatter(record, default_):
        # Check the level and return a specific format string
        format_vals = " " * 33 + "{message}"
        format = format_vals if record["level"].name == "vals" else default_
        return format + "\n"

    format_cli = "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level:<5}</level> | <level>{message}</level>"
    format_file = "{time:YYYY-MM-DD at HH:mm:ss} | {level:<5} | {message}"
    logger.add(
        sys.stderr,
        format=lambda x: custom_formatter(x, format_cli),
        level=log_level_cli,
        colorize=True,
    )
    if file_name is not None:
        logger.add(
            os.path.join(path_to_log_folder, file_name),
            format=lambda x: custom_formatter(x, format_file),
            level=log_level_file,
            mode="w",
            colorize=True,
        )
    logger.add(
        os.path.join(path_to_log_folder, "00_infos.log"),
        format=lambda x: custom_formatter(x, format_file),
        level=20,
        mode="w",
        colorize=True,
    )

    logger.info(f"Logging is set up: logs will be saved to {path_to_log_folder}")


def log_results(
    results: dict,
    level: str = "vals",
    keys: list = [
        "k",
        "loss_to_A",
        "loss_to_theta",
        "loss_to_A_det",
        "loss_to_theta_det",
        "k_det",
        "k_max",
    ],
    all=False,
):
    logger.debug(f"Completed n = {results['n']}, rep = {results['rep']:<4}" + 18 * "-")
    if all:
        logger.log(level, f"{json.dumps(results, indent=4, default=str)}")
    else:
        for key, value in results.items():
            if key in keys:
                logger.log(level, f"{key:<20}: {value:.4f}")


def check_gpu_memory():
    """
    Scans all objects in memory to find tensors on the GPU.  Returns: A list of (Tensor
    Size, Memory Usage in GiB)
    """
    total_mem = 0
    # Header: MB -> GiB
    logger.debug(f"{'Shape':<20} | {'Dtype':<10} | {'Size (GiB)':<10}")
    logger.debug("-" * 45)

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if obj.is_cuda:
                    numel = obj.numel()
                    element_size = obj.element_size()

                    # Math: 1024**2 (MB) -> 1024**3 (GiB)
                    mem_gib = (numel * element_size) / (1024**3)

                    logger.debug(
                        f"{str(list(obj.size())):<20} | {str(obj.dtype):<10} | {mem_gib:.4f}"
                    )
                    total_mem += mem_gib
        except Exception:
            pass

    logger.debug("-" * 45)
    logger.debug(f"Total Tensors Memory: {total_mem:.4f} GiB")
    logger.debug(
        f"Total Reserved Memory: {torch.cuda.memory_reserved() / (1024**3):.4f} GiB"
    )
