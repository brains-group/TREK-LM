import argparse
import json
import os
import warnings
from typing import Dict, Tuple

import flwr as fl
from flwr.common import Context, Parameters
from flwr.common.typing import NDArrays
from peft import PeftModel
from transformers import AutoModelForCausalLM

from utils.data import load_federated_dataset
from utils.models import get_tokenizer_and_data_collator, get_model
from utils.training import (
    backend_setup,
    fit_weighted_average,
    gen_client_fn,
    get_evaluate_fn,
    get_on_fit_config,
    set_seed,
)
from utils.utils import (
    parse_args_with_config,
    print_config,
    generate_deterministic_run_name,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_initial_parameters(model_cfg, checkpoint_path: str) -> Parameters:
    """Load initial parameters from a checkpoint."""
    model = get_model(model_cfg)
    model.load_adapter(checkpoint_path)

    state_dict = model.state_dict()
    nd_arrays = [val.cpu().numpy() for _, val in state_dict.items()]
    return fl.common.ndarrays_to_parameters(nd_arrays)


def main():
    """
    Main function to run a federated training experiment using Flower.

    This function orchestrates the entire process:
    1. Parses command-line arguments for configuration.
    2. Sets the random seed for reproducibility.
    3. Generates a deterministic run name and save path.
    4. Checks for existing state to resume training.
    5. Loads the federated dataset and tokenizer.
    6. Configures the Flower client app and server app.
    7. Starts the Flower simulation.
    8. Marks the training as complete upon finishing.
    """
    cfg, original_cfg = parse_args_with_config()
    print("Configuration:")
    print_config(cfg)

    set_seed(cfg.seed)
    print(f"Using seed: {cfg.seed}")

    if "training_arguments" in cfg.train:
        cfg.train.training_arguments.seed = cfg.seed

    # Determine run name and save path
    run_name = generate_deterministic_run_name(cfg, original_cfg)
    save_path = f"./models/federated/{run_name}"

    # Check if training is already complete
    if os.path.exists(os.path.join(save_path, "training_complete.txt")):
        print(
            f"Training already complete for this configuration. Skipping. Path: {save_path}"
        )
        return

    os.makedirs(save_path, exist_ok=True)

    # Check for server state to resume
    initial_parameters = None
    start_round = 1
    server_state_path = os.path.join(save_path, "server_state.json")
    if os.path.exists(server_state_path):
        with open(server_state_path, "r") as f:
            state = json.load(f)
            last_round = state["last_round"]
            print(f"Resuming from round {last_round + 1}")
            start_round = last_round + 1
            checkpoint_path = os.path.join(save_path, f"checkpoint-{last_round}")
            if os.path.exists(checkpoint_path):
                initial_parameters = get_initial_parameters(cfg.model, checkpoint_path)
            else:
                print(
                    f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch."
                )
                start_round = 1

    # Load federated dataset and tokenizer
    dataset_path = cfg.dataset.path.format(cfg.dataset.name)
    datasets = load_federated_dataset(dataset_path)
    cfg.flower.num_clients = len(datasets)

    # Load dataset and tokenizer
    tokenizer, _ = get_tokenizer_and_data_collator(
        cfg.model.name,
        cfg.model.use_fast_tokenizer,
        cfg.train.padding_side,
        cfg.model.response_template,
    )

    client_app = fl.client.ClientApp(
        client_fn=gen_client_fn(datasets, tokenizer, cfg.model, cfg.train, save_path)
    )

    def server_fn(context: Context):
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=cfg.flower.num_clients,
            fraction_fit=cfg.flower.sample_clients / cfg.flower.num_clients,
            fraction_evaluate=0.0,
            on_fit_config_fn=get_on_fit_config(),
            fit_metrics_aggregation_fn=fit_weighted_average,
            evaluate_fn=get_evaluate_fn(
                cfg.model, cfg.train.save_every_round, cfg.flower.num_rounds, save_path
            ),
            initial_parameters=initial_parameters,
        )
        server_config = fl.server.ServerConfig(
            num_rounds=cfg.flower.num_rounds,
        )
        # Flower's server does not have a concept of start_round, so the simulation
        # will run for num_rounds, but the evaluate_fn will save checkpoints
        # with the correct global round number if we were to adjust it.
        # For simplicity, we let it run and it will overwrite checkpoints if resuming.
        # A more robust solution would involve a custom server/strategy.
        return fl.server.ServerAppComponents(strategy=strategy, config=server_config)

    fl.simulation.run_simulation(
        server_app=fl.server.ServerApp(server_fn=server_fn),
        client_app=client_app,
        num_supernodes=cfg.flower.num_clients,
        backend_config={
            "client_resources": dict(cfg.flower.client_resources),
            "init_args": backend_setup,
        },
    )

    # Mark training as complete
    with open(os.path.join(save_path, "training_complete.txt"), "w") as f:
        f.write("complete")
    print(f"Federated training complete. Final model available at: {save_path}")


if __name__ == "__main__":
    main()
