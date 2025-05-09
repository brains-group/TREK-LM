import numpy as np
import torch

from torch.utils.data import DataLoader

from models import KGEModel, ModE, HAKE

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

import flwr as fl
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar

import random
from datetime import datetime
import argparse
import os
import json
import logging
import sys
import time
import pickle
from typing import Callable, Dict, Tuple, List
from collections import OrderedDict
from logging import ERROR


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training and Testing Knowledge Graph Embedding Models",
        usage="runs.py [<args>] [-h | --help]",
    )

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--data_path", type=str, default="../data/FB15k-237")
    parser.add_argument("--model", default="HAKE", type=str)

    parser.add_argument("-n", "--negative_sample_size", default=256, type=int)
    parser.add_argument("-d", "--hidden_dim", default=1000, type=int)
    parser.add_argument("-g", "--gamma", default=9.0, type=float)
    parser.add_argument("-a", "--adversarial_temperature", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument(
        "--test_batch_size", default=4, type=int, help="valid/test batch size"
    )
    parser.add_argument("-mw", "--modulus_weight", default=3.5, type=float)
    parser.add_argument("-pw", "--phase_weight", default=1.0, type=float)

    parser.add_argument("-lr", "--learning_rate", default=0.00005, type=float)
    parser.add_argument("-cpu", "--cpu_num", default=10, type=int)
    parser.add_argument("-init", "--init_checkpoint", default=None, type=str)
    parser.add_argument("-save", "--save_path", default=None, type=str)
    parser.add_argument("--max_steps", default=10000, type=int)

    parser.add_argument("--save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("--valid_steps", default=10000, type=int)
    parser.add_argument(
        "--log_steps", default=100, type=int, help="train log every xx steps"
    )
    parser.add_argument(
        "--test_log_steps", default=1000, type=int, help="valid/test log every xx steps"
    )

    parser.add_argument(
        "--no_decay", action="store_true", help="Learning rate do not decay"
    )

    # fed args
    parser.add_argument(
        "--num_rounds",
        type=float,
        default=50,
        help="Dropout probability for convolution layer",
    )
    parser.add_argument(
        "--sample_clients",
        type=float,
        default=10,
        help="Dropout probability for convolution layer",
    )
    parser.add_argument(
        "--num_clients",
        type=float,
        default=20,
        help="Dropout probability for convolution layer",
    )

    return parser.parse_args(args)


def override_config(args):
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as f:
        args_dict = json.load(f)

    args.model = args_dict["model"]
    args.data_path = args_dict["data_path"]
    args.hidden_dim = args_dict["hidden_dim"]
    args.test_batch_size = args_dict["test_batch_size"]


def save_model(model, optimizer, save_variable_list, args):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """

    args_dict = vars(args)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    torch.save(
        {
            **save_variable_list,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(args.save_path, "checkpoint"),
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "entity_embedding"), entity_embedding)

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "relation_embedding"), relation_embedding)


def set_logger(args):
    """
    Write logs to checkpoint and console
    """

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, "train.log")
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, "test.log")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(self, args, data_reader, save_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data_reader = data_reader
        num_entity = len(self.data_reader.entity_dict)
        num_relation = len(self.data_reader.relation_dict)

        self.kge_model = HAKE(
            num_entity,
            num_relation,
            args.hidden_dim,
            args.gamma,
            args.modulus_weight,
            args.phase_weight,
        )

        self.kge_model = self.kge_model.cuda()

        train_dataloader_head = DataLoader(
            TrainDataset(
                self.data_reader, args.negative_sample_size, BatchType.HEAD_BATCH
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(
                self.data_reader, args.negative_sample_size, BatchType.TAIL_BATCH
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn,
        )

        self.train_iterator = BidirectionalOneShotIterator(
            train_dataloader_head, train_dataloader_tail
        )

        # Set training configuration
        self.current_learning_rate = args.learning_rate
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),
            lr=self.current_learning_rate,
        )

        self.warm_up_steps = args.max_steps // 2

        self.save_path = save_path

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        parameters = [
            val.cpu().numpy() for _, val in self.kge_model.state_dict().items()
        ]
        return parameters

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        set_parameters(self.kge_model, parameters)

        init_step = 0
        for step in range(init_step, self.args.max_steps):

            self.kge_model.train_step(
                self.kge_model, self.optimizer, self.train_iterator, self.args
            )

            if step >= self.warm_up_steps:
                if not self.args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.kge_model.parameters()),
                    lr=current_learning_rate,
                )
                self.warm_up_steps = self.warm_up_steps * 3

        metrics = {}
        metrics = {**metrics, "train_loss": 0}

        return (
            self.get_parameters({}),
            len(self.data_reader.train_data),
            metrics,
        )


def gen_client_fn(args, save_path) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        data_reader = DataReader(
            args.data_path, int(context.node_config["partition-id"])
        )
        return FlowerClient(args, data_reader, save_path).to_client()

    return client_fn


# Server Stuff
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def main(args):

    args.save_path = f"../../models/KBGAT/{args.data_path.split("/")[-1]}/{(datetime.now()).strftime("%Y%m%d%H%M%S")}"
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    client = fl.client.ClientApp(
        client_fn=gen_client_fn(args, args.save_path),
        # mods=[fixedclipping_mod] # For Differential Privacy
    )

    # Get function that will be executed by the strategy's evaluate() method
    # Here we use it to save global model checkpoints
    def get_evaluate_fn(args, save_every_round, total_round, save_path):
        """Return an evaluation function for saving global model."""

        def evaluate(server_round: int, parameters, config):
            # Save model
            if server_round != 0 and (
                server_round == total_round or server_round % save_every_round == 0
            ):
                entity_embeddings, relation_embeddings = init_embeddings(
                    os.path.join(args.data, "entity2vec.txt"),
                    os.path.join(args.data, "relation2vec.txt"),
                )
                entity_embeddings = torch.FloatTensor(entity_embeddings)
                relation_embeddings = torch.FloatTensor(relation_embeddings)

                data_reader = DataReader(args.data_path)
                num_entity = len(data_reader.entity_dict)
                num_relation = len(data_reader.relation_dict)

                # Init model
                model = HAKE(
                    num_entity,
                    num_relation,
                    args.hidden_dim,
                    args.gamma,
                    args.modulus_weight,
                    args.phase_weight,
                )
                set_parameters(model, parameters)

                path = f"{save_path}/peft_{server_round}/"
                os.makedirs(path, exist_ok=True)
                path += f"trained_{server_round}.pth"
                torch.save(model.state_dict(), path)
                save_model(model, torch.optim.Adam(), {}, args)

            return 0.0, {}

        return evaluate

    def server_fn(context: Context):

        # Define the Strategy
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=args.num_clients,  # total clients
            fraction_fit=args.sample_clients
            / args.num_clients,  # ratio of clients to sample
            fraction_evaluate=0.0,  # No federated evaluation
            # A (optional) function used to configure a "fit()" round
            on_fit_config_fn=get_on_fit_config(),
            # A (optional) function to aggregate metrics sent by clients
            fit_metrics_aggregation_fn=fit_weighted_average,
            # A (optional) function to execute on the server after each round.
            # In this example the function only saves the global model.
            evaluate_fn=get_evaluate_fn(args, 5, args.num_rounds, args.save_path),
        )

        # # Add Differential Privacy
        # sampled_clients = cfg.flower.num_clients*strategy.fraction_fit
        # strategy = DifferentialPrivacyClientSideFixedClipping(
        #     strategy,
        #     noise_multiplier=cfg.flower.dp.noise_mult,
        #     clipping_norm=cfg.flower.dp.clip_norm,
        #     num_sampled_clients=sampled_clients
        # )

        # Number of rounds to run the simulation
        num_rounds = args.num_rounds
        config = fl.server.ServerConfig(num_rounds=num_rounds)

        return fl.server.ServerAppComponents(strategy=strategy, config=config)

    server = fl.server.ServerApp(server_fn=server_fn)

    client_resources = {"num_cpus": 8, "num_gpus": 1.0}
    fl.simulation.run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": client_resources,
            "init_args": {"logging_level": ERROR, "log_to_driver": False},
        },
        verbose_logging=True,
    )


if __name__ == "__main__":
    main(parse_args())
