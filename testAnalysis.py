import json
import os
import random
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

from tqdm import tqdm

# Import the main functions from the scripts to be called
from train_centralized import main as train_main
from test import main as test_main
from utils.utils import generate_deterministic_run_name, parse_args_with_config


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


def main():
    """
    Main function to run the test analysis pipeline.
    This script samples users, trains a model for each, runs evaluation,
    and aggregates the results.
    """
    random.seed(2025)

    # Load the test dataset to identify users
    with open("./data/movieKnowledgeGraphTestDataset.json", "r") as file:
        dataset = json.load(file)

    user_counts = defaultdict(int)
    for datapoint in dataset:
        user_match = re.search(
            r"The user's entity is represented by (\d+).",
            datapoint["prompt"][0]["content"],
        )
        if user_match:
            user_counts[user_match.group(1)] += 1

    # Sample users
    eligible_users = [user for user, count in user_counts.items() if count >= 10]
    selected_users = random.sample(eligible_users, min(10, len(eligible_users)))
    print(f"Selected users for analysis: {selected_users}")

    # Initialize result aggregators
    real_metrics_agg = defaultdict(float)
    synth_metrics_agg = defaultdict(float)

    for user_id in tqdm(selected_users, desc="Analyzing Users"):
        print(f"\n----- Processing User: {user_id} -----")

        # --- Real Data Training and Testing ---
        print(f"Training on real data for User: {user_id}")
        train_argv_real = [
            "--cfg",
            "conf/centralized_full.yaml",
            "--dataset.name",
            "movieKnowledgeGraphDataset",
            "--dataset_index",
            user_id,
        ]
        run_in_memory(train_main, train_argv_real)

        print(f"Testing on real data for User: {user_id}")
        lora_path_real = get_run_path(train_argv_real)
        latest_checkpoint_real = max(os.listdir(lora_path_real))
        test_argv_real = [
            "--lora_path",
            os.path.join(lora_path_real, latest_checkpoint_real),
            "--user_id",
            user_id,
        ]
        metrics_real, _ = run_in_memory(test_main, test_argv_real)
        for key, value in metrics_real.items():
            real_metrics_agg[key] += value

        # --- Synthetic Data Training and Testing ---
        print(f"Training on synthetic data for User: {user_id}")
        train_argv_synth = [
            "--cfg",
            "conf/centralized_full.yaml",
            "--dataset.name",
            "movieKnowledgeGraphDatasetWithSyntheticData",
            "--dataset_index",
            user_id,
        ]
        run_in_memory(train_main, train_argv_synth)

        print(f"Testing on synthetic data for User: {user_id}")
        lora_path_synth = get_run_path(train_argv_synth)
        latest_checkpoint_synth = max(os.listdir(lora_path_synth))
        test_argv_synth = [
            "--lora_path",
            os.path.join(lora_path_synth, latest_checkpoint_synth),
            "--user_id",
            user_id,
        ]
        metrics_synth, _ = run_in_memory(test_main, test_argv_synth)
        for key, value in metrics_synth.items():
            synth_metrics_agg[key] += value

    # Print final aggregated results
    num_selected_users = len(selected_users)
    print("\n" + "=" * 40)
    print(f"      Aggregated Analysis Results ({num_selected_users} users)")
    print("=" * 40)

    if num_selected_users > 0:
        print("\n--- Real Data Average Metrics ---")
        for key, value in sorted(real_metrics_agg.items()):
            print(f"{key}: {value / num_selected_users:.4f}")

        print("\n--- Synthetic Data Average Metrics ---")
        for key, value in sorted(synth_metrics_agg.items()):
            print(f"{key}: {value / num_selected_users:.4f}")

    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
