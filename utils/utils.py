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
import os
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize
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


def get_config(config_name: str, overrides: List[str] = None):
    with initialize(config_path="../conf", version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


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


def parse_args_with_config() -> tuple[DictConfig, DictConfig]:
    """
    Parses command-line arguments, loading a config file and allowing overrides.
    Returns the final, merged config and the original config from the file.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to YAML config file."
    )
    args, remaining_argv = parser.parse_known_args()

    # Load the configuration using the custom get_config, applying command-line overrides
    cfg = get_config(args.cfg, remaining_argv)

    # The original_cfg is the config without command-line overrides.
    # We can get this by calling get_config again without overrides.
    original_cfg = get_config(args.cfg)

    # Add config_path to the final config for consistency with previous behavior
    with open_dict(cfg):
        cfg.config_path = args.cfg
    with open_dict(original_cfg):
        original_cfg.config_path = args.cfg

    return cfg, original_cfg


def generate_deterministic_run_name(cfg: DictConfig, original_cfg: DictConfig) -> str:
    """Generates a deterministic run name from the config file and overrides."""
    import os

    config_name = cfg.config_path

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
    filtered_overrides = [
        ov for ov in overrides if not any(ex in ov for ex in exclude_keys)
    ]

    run_name = config_name
    if filtered_overrides:
        # Create a sorted, sanitized string of overrides
        override_str = (
            "-".join(sorted(filtered_overrides)).replace("=", "_").replace(".", "_")
        )
        run_name += "-" + override_str

    return run_name.replace("/", "_")


def compute_communication_costs(config, comm_bw_mbps: float = 20):
    """Computes and prints the communication costs for a given model and FL setting."""
    model = get_model(config.model)

    trainable, all_parameters = model.get_nb_trainable_parameters()

    total_size = 4 * all_parameters / (1024**2)
    trainable_size = 4 * trainable / (1024**2)

    upload_time_total = total_size / (comm_bw_mbps / 8)
    upload_time_finetune = trainable_size / (comm_bw_mbps / 8)

    print(
        f"Full model:\n\t{all_parameters/1e6:.3f} M parameters\n\t{total_size:.2f} MB --> upload in {upload_time_total:.2f}s @ {comm_bw_mbps}Mbps"
    )
    print(
        f"Finetuned model:\n\t{trainable/1e6:.3f} M parameters\n\t{trainable_size:.2f} MB --> upload in {upload_time_finetune:.2f}s @ {comm_bw_mbps}Mbps"
    )
    # print(f"In a {comm_bw_mbps} Mbps channel --> {}")

    num_rounds = config.flower.num_rounds
    num_clients_per_round = int(config.flower.sample_clients)
    print(
        f"Federated Learning setting: "
        f"\n\tNumber of rounds: {num_rounds}"
        f"\n\tNumber of clients per round: {num_clients_per_round}"
    )

    print(f"-----------------------------------------------")
    print(
        f"Total Communication costs (Full model): {2*num_rounds*num_clients_per_round*total_size/1024:.1f} GB"
    )
    print(
        f"Total Communication costs (Finetuning): {2*num_rounds*num_clients_per_round*trainable_size} MB"
    )
    print(f"Communication savings: {all_parameters/trainable:.1}x")
