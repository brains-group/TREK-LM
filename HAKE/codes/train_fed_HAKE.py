import numpy as np
import torch

from torch.utils.data import DataLoader

from models import KGEModel, ModE, HAKE

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator

from models import SpKBGATModified, SpKBGATConvOnly
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

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model", default="TransE", type=str)

    parser.add_argument("-n", "--negative_sample_size", default=128, type=int)
    parser.add_argument("-d", "--hidden_dim", default=500, type=int)
    parser.add_argument("-g", "--gamma", default=12.0, type=float)
    parser.add_argument("-a", "--adversarial_temperature", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument(
        "--test_batch_size", default=4, type=int, help="valid/test batch size"
    )
    parser.add_argument("-mw", "--modulus_weight", default=1.0, type=float)
    parser.add_argument("-pw", "--phase_weight", default=0.5, type=float)

    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float)
    parser.add_argument("-cpu", "--cpu_num", default=10, type=int)
    parser.add_argument("-init", "--init_checkpoint", default=None, type=str)
    parser.add_argument("-save", "--save_path", default=None, type=str)
    parser.add_argument("--max_steps", default=100000, type=int)

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

    def __init__(self, args, save_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        data_reader = DataReader(args.data_path)
        num_entity = len(data_reader.entity_dict)
        num_relation = len(data_reader.relation_dict)

        if args.model == "ModE":
            self.kge_model = ModE(num_entity, num_relation, args.hidden_dim, args.gamma)
        elif args.model == "HAKE":
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
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
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

        set_parameters(self.model_conv, parameters)

        init_step = 0
        step = init_step

        # Training Loop
        for step in range(init_step, self.args.max_steps):

            self.kge_model.train_step(
                self.kge_model, optimizer, self.train_iterator, args
            )

            if step >= warm_up_steps:
                if not self.args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.kge_model.parameters()),
                    lr=current_learning_rate,
                )
                warm_up_steps = warm_up_steps * 3

            if step % self.args.save_checkpoint_steps == 0:
                save_variable_list = {
                    "step": step,
                    "current_learning_rate": current_learning_rate,
                    "warm_up_steps": warm_up_steps,
                }
                save_model(self.kge_model, optimizer, save_variable_list, self.args)

        metrics = {}
        metrics = {**metrics, "train_loss": sum(epoch_losses) / len(epoch_losses)}

        return (
            self.get_parameters({}),
            len(self.Corpus_.train_triples),
            metrics,
        )


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError("one of train/val/test mode must be choosed.")

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError("one of init_checkpoint/data_path must be choosed.")

    if args.do_train and args.save_path is None:
        raise ValueError("Where do you want to save your trained model?")

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


if __name__ == "__main__":
    main(parse_args())
