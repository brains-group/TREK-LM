"""
A module for general-purpose utility functions, including configuration management,
logging, and result visualization.
"""
import argparse
import logging
import textwrap
from logging import LogRecord
from typing import List
import yaml

import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from omegaconf import DictConfig, OmegaConf
from flwr.common.logger import (
    ConsoleHandler,
    console_handler,
    FLOWER_LOGGER,
    LOG_COLORS,
)

from .models import get_model


def format_string(msg, char_width: int = 50) -> str:
    """Wraps a string to a given character width."""
    return textwrap.fill(msg, char_width, subsequent_indent="\t")


def print_config(config: DictConfig):
    """Prints the config as YAML."""
    print(OmegaConf.to_yaml(config))


# Remove default Flower logger handler to replace it
FLOWER_LOGGER.removeHandler(console_handler)


class ConsoleHandlerV2(ConsoleHandler):
    """A console handler with more compact logging format."""
    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Configure and add the custom console logger
console_handlerv2 = ConsoleHandlerV2(timestamps=False, json=False, colored=True)
console_handlerv2.setLevel(logging.INFO)
FLOWER_LOGGER.addHandler(console_handlerv2)


def get_config(config_path: str) -> DictConfig:
    """Loads a YAML config file into a DictConfig object."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return OmegaConf.create(cfg)


def _add_args_from_config(parser, config, parent_key=""):
    """Recursively add arguments to the parser from the config using dot notation."""
    for key, value in config.items():
        arg_name = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, DictConfig):
            _add_args_from_config(parser, value, parent_key=arg_name)
        else:
            # Handle cases where value is None
            arg_type = type(value) if value is not None else str
            parser.add_argument(f"--{arg_name}", type=arg_type, default=value)


def parse_args_with_config() -> (DictConfig, DictConfig):
    """
    Parses command-line arguments, loading a config file and allowing overrides.
    Returns the final, merged config and the original config from the file.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file.")
    args, remaining_argv = parser.parse_known_args()

    original_cfg = get_config(args.cfg)
    cfg = original_cfg.copy()

    full_parser = argparse.ArgumentParser()
    _add_args_from_config(full_parser, cfg)
    full_parser.add_argument("--cfg", type=str, default=args.cfg)
    final_args = full_parser.parse_args(remaining_argv)

    override_cfg = OmegaConf.create(vars(final_args))
    cfg = OmegaConf.merge(cfg, override_cfg)
    cfg.config_path = args.cfg

    return cfg, original_cfg


def generate_deterministic_run_name(cfg: DictConfig, original_cfg: DictConfig) -> str:
    """Generates a deterministic run name from the config file and overrides."""
    import os
    config_name = os.path.splitext(os.path.basename(cfg.config_path))[0]

    overrides = []

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    flat_original_cfg = OmegaConf.to_container(original_cfg, resolve=True)

    # Recursively find differences between the final config and the original
    def find_overrides(d1, d2, prefix=""):
        for k, v1 in d1.items():
            key_path = f"{prefix}.{k}" if prefix else k
            if k not in d2 or v1 != d2.get(k):
                if isinstance(v1, dict) and isinstance(d2.get(k), dict):
                    find_overrides(v1, d2[k], prefix=key_path)
                else:
                    overrides.append(f"{key_path}={v1}")

    find_overrides(flat_cfg, flat_original_cfg)

    # Filter out keys that are not useful for the run name
    exclude_keys = ["config_path", "dataset.path"]
    filtered_overrides = [ov for ov in overrides if not any(ex in ov for ex in exclude_keys)]

    run_name = config_name
    if filtered_overrides:
        # Create a sorted, sanitized string of overrides
        override_str = "-".join(sorted(filtered_overrides)).replace("=", "_").replace(".", "_")
        run_name += "-" + override_str

    return run_name.replace("/", "_")


def visualize_partitions(fed_dataset: FederatedDataset):
    """Visualizes the number of examples in each partition of a FederatedDataset."""
    #... (implementation remains the same)

def compute_communication_costs(config, comm_bw_mbps: float = 20):
    """Computes and prints the communication costs for a given model and FL setting."""
    #... (implementation remains the same)

#... (plotting functions remain the same)
