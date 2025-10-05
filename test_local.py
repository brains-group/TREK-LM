import argparse
import json
import os
import random
import re
import sys
import numpy as np
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

from tqdm import tqdm

# Import the main functions from the scripts to be called
from train_centralized import main as train_main
from test import main as test_main
from utils.utils import (
    generate_deterministic_run_name,
    get_config,
    parse_args_with_config,
)
from utils.evaluation import save_metrics_to_csv


def get_run_path(argv):
    """
    Determines the save path for a training run based on command-line arguments.
    """
    original_argv = sys.argv
    # The first argument is the script name, which is not used by parse_args_with_config
    # but is expected to be present.
    sys.argv = ["train_centralized.py"] + argv
    try:
        cfg, original_cfg = parse_args_with_config()
        run_name = generate_deterministic_run_name(cfg, original_cfg)
        save_path = f"./models/centralized/{run_name}"
    finally:
        sys.argv = original_argv
    return save_path


def run_in_memory(target_main, argv):
    """
    Runs a script's main function in memory with specified command-line arguments.

    This function temporarily replaces `sys.argv` to simulate a command-line
    execution of another script's main function. It captures and returns any
    output printed to stdout, along with the function's return value.

    Args:
        target_main: The main function to execute (e.g., train_main).
        argv: A list of strings representing the command-line arguments.

    Returns:
        A tuple containing the return value of the target main function and
        a string with the captured stdout.
    """
    original_argv = sys.argv
    sys.argv = [target_main.__module__] + argv

    stdout_capture = StringIO()
    try:
        with redirect_stdout(stdout_capture):
            result = target_main()
    finally:
        sys.argv = original_argv

    output = stdout_capture.getvalue()
    return result, output


def calculate_and_format_metrics(aggregated_metrics, num_selected_users):
    final_metrics = {}

    final_metrics["num_selected_users"] = num_selected_users

    for key, values in aggregated_metrics.items():
        # Skip non-numeric metrics or those that are already standard deviation/error
        if (
            not isinstance(values, list)
            or not values
            or not (isinstance(values[0], float) or values[0] == 0)
            or "_SE" in key
            or "_StdDev" in key
        ):
            continue

        mean_val = np.mean(values)
        final_metrics[key] = mean_val

        # Calculate standard deviation for all metrics when aggregating across users
        final_metrics[f"{key}_StdDev"] = np.std(values, ddof=1)

    sorted_keys = sorted(
        final_metrics.keys(),
        key=lambda x: (
            x.replace("_StdDev", ""),
            "_StdDev" if "_StdDev" in x else "",
        ),
    )
    for key in sorted_keys:
        value = final_metrics[key]
        print(f"{key}: {value:.4f}")

    return final_metrics


def main():
    """
    Main function to run the test analysis pipeline.
    This script samples users, trains a model for each, runs evaluation,
    and aggregates the results.
    """
    parser = argparse.ArgumentParser(
        description="Analyze model performance across selected users."
    )
    parser.add_argument(
        "--min_datapoints",
        type=int,
        default=10,
        help="Minimum number of test datapoints for a user to be eligible.",
    )
    parser.add_argument(
        "--num_selected_users",
        type=int,
        default=10,
        help="Number of eligible users to select for analysis.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="centralized_Qwen3-0.6B_movie_3:4",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="movieKnowledgeGraphDataset",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="movieKnowledgeGraphTestDataset",
    )
    parser.add_argument(
        "--num_test_datapoints",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    random.seed(2025)

    # Load the test dataset to identify users
    with open(f"./data/{args.test_data}.json", "r") as file:
        dataset = json.load(file)
    with open(f"./data/{args.data}.json", "r") as file:
        trainDataset = json.load(file)

    user_counts = defaultdict(int)
    for datapoint in dataset:
        user_match = re.search(
            r"The user's entity is represented by /?(\d+).",
            datapoint["prompt"][0]["content"],
        )
        if user_match:
            user_counts[user_match.group(1)] += 1
    trainUsers = defaultdict(bool)
    for user in trainDataset.keys():
        trainUsers[user] = True

    # Sample users
    eligible_users = [
        user
        for user, count in user_counts.items()
        if ((count >= args.min_datapoints) and trainUsers[user])
    ]
    selected_users = random.sample(
        eligible_users, min(args.num_selected_users, len(eligible_users))
    )
    print(f"Selected users for analysis: {selected_users}")

    # Initialize result aggregators
    real_metrics_agg = defaultdict(list)
    synth_metrics_agg = defaultdict(list)

    config = get_config(args.cfg)

    for user_id in tqdm(selected_users, desc="Analyzing Users"):
        print(f"\n----- Processing User: {user_id} -----")

        # --- Real Data Training and Testing ---
        print(f"Training on real data for User: {user_id}")
        train_argv_real = [
            "--cfg",
            args.cfg,
            f'dataset.name="{args.data}"',
            f'+dataset_index="{user_id}"',
        ]
        run_in_memory(train_main, train_argv_real)

        print(f"Testing on real data for User: {user_id}")
        lora_path_real = get_run_path(train_argv_real)
        latest_checkpoint_real = max(os.listdir(lora_path_real))
        test_argv_real = [
            "--base_model_path",
            config.model.name,
            "--lora_path",
            # os.path.join(lora_path_real, latest_checkpoint_real),
            lora_path_real,
            "--user_id",
            user_id,
            "--data_path",
            f"./data/{args.test_data}.json",
        ]
        if args.num_test_datapoints:
            test_argv_real += ["--max_datapoints", str(args.num_test_datapoints)]
        metrics_real, _ = run_in_memory(test_main, test_argv_real)
        for key, value in metrics_real.items():
            real_metrics_agg[key].append(value)

        if "movie" in args.data.lower():
            # --- Synthetic Data Training and Testing ---
            print(f"Training on synthetic data for User: {user_id}")
            train_argv_synth = [
                "--cfg",
                args.cfg,
                f'dataset.name="{args.data}WithSyntheticData"',
                f'+dataset_index="{user_id}"',
            ]
            run_in_memory(train_main, train_argv_synth)

            print(f"Testing on synthetic data for User: {user_id}")
            lora_path_synth = get_run_path(train_argv_synth)
            latest_checkpoint_synth = max(os.listdir(lora_path_synth))
            test_argv_synth = [
                "--base_model_path",
                config.model.name,
                "--lora_path",
                # os.path.join(lora_path_synth, latest_checkpoint_synth),
                lora_path_real,
                "--user_id",
                user_id,
                "--data_path",
                f"./data/{args.test_data}.json",
            ]
            if args.num_test_datapoints:
                test_argv_synth += ["--max_datapoints", str(args.num_test_datapoints)]
            metrics_synth, _ = run_in_memory(test_main, test_argv_synth)
            for key, value in metrics_synth.items():
                synth_metrics_agg[key].append(value)

    # Print final aggregated results
    num_selected_users = len(selected_users)
    print("\n" + "=" * 40)
    print(f"      Aggregated Analysis Results ({num_selected_users} users)")
    print("=" * 40)

    if num_selected_users > 0:
        print("\n--- Real Data Average Metrics ---")
        real_final_metrics = calculate_and_format_metrics(
            real_metrics_agg, args.num_selected_users
        )
        save_metrics_to_csv(
            real_final_metrics,
            f"./metrics/aggregated_real_data_{args.cfg}_{args.num_selected_users}_{args.data}.csv",
            True,
        )
        if "movie" in args.data.lower():
            print("\n--- Synthetic Data Average Metrics ---")
            synth_final_metrics = calculate_and_format_metrics(
                synth_metrics_agg, args.num_selected_users
            )
            save_metrics_to_csv(
                synth_final_metrics,
                f"./metrics/aggregated_synthetic_data_{args.cfg}_{args.num_selected_users}_{args.data}.csv",
                True,
            )

    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
