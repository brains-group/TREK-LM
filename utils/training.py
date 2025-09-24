import math
from collections import OrderedDict
from logging import ERROR
from typing import Callable, Dict, Tuple

import flwr as fl
import torch
from datasets import Dataset
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from peft import get_peft_model_state_dict
from trl import KTOConfig, KTOTrainer

import random
import numpy as np
from .models import get_model, set_parameters

# To filter out all warnings from HF that come from the client side
backend_setup = {"logging_level": ERROR, "log_to_driver": False}


def calculate_kto_weights(
    dataset, desirable_prompt_weight: float = 3.0, undesirable_prompt_weight: float = 4.0
) -> Tuple[float, float]:
    """Calculates weights for KTO loss based on dataset composition."""
    desirable_weight, undesirable_weight = 1.0, 1.0
    if dataset:
        num_desirable_samples = sum(dataset["label"])
        num_undesirable_samples = len(dataset) - num_desirable_samples

        num_desirable = num_desirable_samples * desirable_prompt_weight
        num_undesirable = num_undesirable_samples * undesirable_prompt_weight

        if num_desirable > 0 and num_undesirable > 0:
            if num_desirable < num_undesirable:
                desirable_weight = num_undesirable / num_desirable
            else:
                undesirable_weight = num_desirable / num_undesirable
    return desirable_weight, undesirable_weight


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for KTO training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        save_path,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.model = get_model(model_cfg)
        self.trainset = trainset

        desirable_prompt_weight = train_cfg.get("desirable_prompt_weight", 3.0)
        undesirable_prompt_weight = train_cfg.get("undesirable_prompt_weight", 4.0)
        desirable_weight, undesirable_weight = calculate_kto_weights(
            trainset, desirable_prompt_weight, undesirable_prompt_weight
        )
        self.training_arguments = KTOConfig(
            **train_cfg.training_arguments,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current model."""
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )
        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path

        evalset = None
        if self.train_cfg.evaluate_split:
            train_test = self.trainset.train_test_split(test_size=0.1, seed=1234)
            trainset = train_test["train"]
            evalset = train_test["test"]
        else:
            trainset = self.trainset

        trainer = KTOTrainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=trainset,
            eval_dataset=evalset,
        )

        metrics = {}
        if self.train_cfg.evaluate_split:
            eval_res = trainer.evaluate()
            metrics["eval_loss"] = eval_res["eval_loss"]
            print(eval_res)

        results = trainer.train()
        metrics = {**metrics, "train_loss": results.training_loss}

        return self.get_parameters({}), len(self.trainset), metrics


def gen_client_fn(
    datasets,
    tokenizer,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        partition_id = int(context.node_config["partition-id"])
        client_trainset = Dataset.from_list(list(datasets.values())[partition_id])
        return FlowerClient(
            model_cfg,
            train_cfg,
            client_trainset,
            tokenizer,
            save_path,
        ).to_client()

    return client_fn


import os
import json


def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path, start_round=1):
    """
    Return an evaluation function for saving the global model and server state.
    """

    def evaluate(server_round: int, parameters, config):
        # Adjust the server_round to be the global round number
        global_round = server_round + start_round - 1

        if global_round == 0:
            return 0.0, {}

        # Save model checkpoint
        if (
            global_round == total_round
            or global_round % save_every_round == 0
        ):
            model = get_model(model_cfg)
            set_parameters(model, parameters)
            model.save_pretrained(
                os.path.join(save_path, f"checkpoint-{global_round}")
            )

            # Save server state (current round)
            state = {"last_round": global_round}
            with open(os.path.join(save_path, "server_state.json"), "w") as f:
                json.dump(state, f)

            print(f"Saved checkpoint and server state for round {global_round}")

        return 0.0, {}

    return evaluate


def get_on_fit_config(
    fit_config_fn=None, start_round=1
) -> Callable[[int], Dict[str, Scalar]]:
    """Return a function which returns a configuration with the current round."""

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with the current round."""
        # The server_round is the local round number (1, 2, 3,...)
        # The start_round is the global round number where the training starts
        # (e.g., 1 if starting from scratch, 101 if resuming from round 100)
        global_round = server_round + start_round - 1
        config = {"current_round": global_round}
        return config

    return fit_config


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}
