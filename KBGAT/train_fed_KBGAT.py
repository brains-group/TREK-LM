import torch

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

from preprocess import (
    read_entity_from_id,
    read_relation_from_id,
    init_embeddings,
    build_data,
)
from create_batch import Corpus
from utils import save_model

import random
from datetime import datetime
import argparse
import os
import sys
import logging
import time
import pickle
from typing import Callable, Dict, Tuple, List
from collections import OrderedDict
from logging import ERROR


data = "FB15k-237"


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument(
        "-data",
        "--data",
        default=f"./data/{data}/",
        help="data directory",
    )
    args.add_argument(
        "-e_g", "--epochs_gat", type=int, default=45, help="Number of epochs"
    )
    args.add_argument(
        "-e_c", "--epochs_conv", type=int, default=3, help="Number of epochs"
    )
    args.add_argument(
        "-w_gat",
        "--weight_decay_gat",
        type=float,
        default=0.00001,
        help="L2 reglarization for gat",
    )
    args.add_argument(
        "-w_conv",
        "--weight_decay_conv",
        type=float,
        default=0.000001,
        help="L2 reglarization for conv",
    )
    args.add_argument(
        "-pre_emb",
        "--pretrained_emb",
        type=bool,
        default=True,
        help="Use pretrained embeddings",
    )
    args.add_argument(
        "-emb_size",
        "--embedding_size",
        type=int,
        default=50,
        help="Size of embeddings (if pretrained not used)",
    )
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)
    args.add_argument(
        "-outfolder",
        "--output_folder",
        default=f"../models/KBGAT/{data}/{(datetime.now()).strftime("%Y%m%d%H%M%S")}",
        help="Folder name to save the models.",
    )

    # arguments for GAT
    args.add_argument(
        "-b_gat", "--batch_size_gat", type=int, default=1360, help="Batch size for GAT"
    )
    args.add_argument(
        "-neg_s_gat",
        "--valid_invalid_ratio_gat",
        type=int,
        default=2,
        help="Ratio of valid to invalid triples for GAT training",
    )
    args.add_argument(
        "-drop_GAT",
        "--drop_GAT",
        type=float,
        default=0.3,
        help="Dropout probability for SpGAT layer",
    )
    args.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        default=0.2,
        help="LeakyRelu alphs for SpGAT layer",
    )
    args.add_argument(
        "-out_dim",
        "--entity_out_dim",
        type=int,
        nargs="+",
        default=[100, 200],
        help="Entity output embedding dimensions",
    )
    args.add_argument(
        "-h_gat",
        "--nheads_GAT",
        type=int,
        nargs="+",
        default=[2, 2],
        help="Multihead attention SpGAT",
    )
    args.add_argument(
        "-margin", "--margin", type=float, default=1, help="Margin used in hinge loss"
    )

    # arguments for convolution network
    args.add_argument(
        "-b_conv",
        "--batch_size_conv",
        type=int,
        default=16,
        help="Batch size for conv",
    )
    args.add_argument(
        "-alpha_conv",
        "--alpha_conv",
        type=float,
        default=0.2,
        help="LeakyRelu alphas for conv layer",
    )
    args.add_argument(
        "-neg_s_conv",
        "--valid_invalid_ratio_conv",
        type=int,
        default=40,
        help="Ratio of valid to invalid triples for convolution training",
    )
    args.add_argument(
        "-o",
        "--out_channels",
        type=int,
        default=50,
        help="Number of output channels in conv layer",
    )
    args.add_argument(
        "-drop_conv",
        "--drop_conv",
        type=float,
        default=0.3,
        help="Dropout probability for convolution layer",
    )

    # fed args
    args.add_argument(
        "--num_rounds",
        type=float,
        default=50,
        help="Dropout probability for convolution layer",
    )
    args.add_argument(
        "--sample_clients",
        type=float,
        default=10,
        help="Dropout probability for convolution layer",
    )
    args.add_argument(
        "--num_clients",
        type=float,
        default=20,
        help="Dropout probability for convolution layer",
    )

    args = args.parse_args()
    return args


args = parse_args()


def load_data(args, partition=None):
    (
        train_data,
        validation_data,
        test_data,
        entity2id,
        relation2id,
        headTailSelector,
        unique_entities_train,
    ) = build_data(args.data, is_unweigted=False, directed=True, partition=partition)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(
            os.path.join(args.data, "entity2vec.txt"),
            os.path.join(args.data, "relation2vec.txt"),
        )
    else:
        entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)

    corpus = Corpus(
        args,
        train_data,
        validation_data,
        test_data,
        entity2id,
        relation2id,
        headTailSelector,
        args.batch_size_gat,
        args.valid_invalid_ratio_gat,
        unique_entities_train,
        args.get_2hop,
    )

    return (
        corpus,
        torch.FloatTensor(entity_embeddings),
        torch.FloatTensor(relation_embeddings),
    )


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self, args, Corpus_, entity_embeddings, relation_embeddings, save_path
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.Corpus_ = Corpus_
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        if args.get_2hop:
            file = args.data + "/2hop.pickle"
            with open(file, "wb") as handle:
                pickle.dump(
                    Corpus_.node_neighbors_2hop,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        if args.use_2hop:
            print("Opening node_neighbors pickle object")
            file = args.data + "/2hop.pickle"
            with open(file, "rb") as handle:
                self.node_neighbors_2hop = pickle.load(handle)

        self.CUDA = torch.cuda.is_available()

        # instantiate models
        self.model_gat = SpKBGATModified(
            entity_embeddings,
            relation_embeddings,
            args.entity_out_dim,
            args.entity_out_dim,
            args.drop_GAT,
            args.alpha,
            args.nheads_GAT,
        )
        self.model_conv = SpKBGATConvOnly(
            entity_embeddings,
            relation_embeddings,
            args.entity_out_dim,
            args.entity_out_dim,
            args.drop_GAT,
            args.drop_conv,
            args.alpha,
            args.alpha_conv,
            args.nheads_GAT,
            args.out_channels,
        )

        if self.CUDA:
            self.model_conv.cuda()
            self.model_gat.cuda()

        self.save_path = save_path

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        parameters = [
            val.cpu().numpy() for _, val in self.model_conv.state_dict().items()
        ]
        return parameters

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        set_parameters(self.model_conv, parameters)

        def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
            len_pos_triples = int(
                train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1)
            )

            pos_triples = train_indices[:len_pos_triples]
            neg_triples = train_indices[len_pos_triples:]

            pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

            source_embeds = entity_embed[pos_triples[:, 0]]
            relation_embeds = relation_embed[pos_triples[:, 1]]
            tail_embeds = entity_embed[pos_triples[:, 2]]

            x = source_embeds + relation_embeds - tail_embeds
            pos_norm = torch.norm(x, p=1, dim=1)

            source_embeds = entity_embed[neg_triples[:, 0]]
            relation_embeds = relation_embed[neg_triples[:, 1]]
            tail_embeds = entity_embed[neg_triples[:, 2]]

            x = source_embeds + relation_embeds - tail_embeds
            neg_norm = torch.norm(x, p=1, dim=1)

            y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

            loss = gat_loss_func(pos_norm, neg_norm, y)
            return loss

        optimizer = torch.optim.Adam(
            self.model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.5, last_epoch=-1
        )

        gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

        current_batch_2hop_indices = torch.tensor([])
        if args.use_2hop:
            current_batch_2hop_indices = self.Corpus_.get_batch_nhop_neighbors_all(
                args, self.Corpus_.unique_entities_train, self.node_neighbors_2hop
            )

        if self.CUDA:
            current_batch_2hop_indices = Variable(
                torch.LongTensor(current_batch_2hop_indices)
            ).cuda()
        else:
            current_batch_2hop_indices = Variable(
                torch.LongTensor(current_batch_2hop_indices)
            )

        for epoch in range(args.epochs_gat):
            random.shuffle(self.Corpus_.train_triples)
            self.Corpus_.train_indices = np.array(
                list(self.Corpus_.train_triples)
            ).astype(np.int32)

            self.model_gat.train()  # getting in training mode

            if len(self.Corpus_.train_indices) % args.batch_size_gat == 0:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // args.batch_size_gat
                )
            else:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // args.batch_size_gat
                ) + 1

            for iters in range(num_iters_per_epoch):
                train_indices, train_values = self.Corpus_.get_iteration_batch(iters)

                if self.CUDA:
                    train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                    train_values = Variable(torch.FloatTensor(train_values)).cuda()
                else:
                    train_indices = Variable(torch.LongTensor(train_indices))
                    train_values = Variable(torch.FloatTensor(train_values))

                # forward pass
                entity_embed, relation_embed = self.model_gat(
                    self.Corpus_,
                    self.Corpus_.train_adj_matrix,
                    train_indices,
                    current_batch_2hop_indices,
                )

                optimizer.zero_grad()

                loss = batch_gat_loss(
                    gat_loss_func, train_indices, entity_embed, relation_embed
                )

                loss.backward()
                optimizer.step()

            scheduler.step()

        self.model_conv.final_entity_embeddings = self.model_gat.final_entity_embeddings
        self.model_conv.final_relation_embeddings = (
            self.model_gat.final_relation_embeddings
        )

        self.Corpus_.batch_size = args.batch_size_conv
        self.Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

        optimizer = torch.optim.Adam(
            self.model_conv.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay_conv,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.5, last_epoch=-1
        )

        margin_loss = torch.nn.SoftMarginLoss()

        epoch_losses = []
        for epoch in range(args.epochs_conv):
            random.shuffle(self.Corpus_.train_triples)
            self.Corpus_.train_indices = np.array(
                list(self.Corpus_.train_triples)
            ).astype(np.int32)

            self.model_conv.train()  # getting in training mode
            epoch_loss = []

            if len(self.Corpus_.train_indices) % args.batch_size_conv == 0:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // args.batch_size_conv
                )
            else:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // args.batch_size_conv
                ) + 1

            for iters in range(num_iters_per_epoch):
                train_indices, train_values = self.Corpus_.get_iteration_batch(iters)

                if self.CUDA:
                    train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                    train_values = Variable(torch.FloatTensor(train_values)).cuda()
                else:
                    train_indices = Variable(torch.LongTensor(train_indices))
                    train_values = Variable(torch.FloatTensor(train_values))

                preds = self.model_conv(
                    self.Corpus_, self.Corpus_.train_adj_matrix, train_indices
                )

                optimizer.zero_grad()

                loss = margin_loss(preds.view(-1), train_values.view(-1))

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.data.item())

            scheduler.step()
            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        metrics = {}
        metrics = {**metrics, "train_loss": sum(epoch_losses) / len(epoch_losses)}

        return (
            self.get_parameters({}),
            len(self.Corpus_.train_triples),
            metrics,
        )


# Defining FlowerClient for KBGAT
def gen_client_fn(args, save_path) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Let's get the partition corresponding to the i-th client
        partition = {
            "partitionID": int(context.node_config["partition-id"]),
            "numPartitions": args.num_clients,
        }
        Corpus_, entity_embeddings, relation_embeddings = load_data(args, partition)
        return FlowerClient(
            args, Corpus_, entity_embeddings, relation_embeddings, save_path
        ).to_client()

    return client_fn


save_path = f"../models/KBGAT/{data}/{(datetime.now()).strftime("%Y%m%d%H%M%S")}"
client = fl.client.ClientApp(
    client_fn=gen_client_fn(args, save_path),
    # mods=[fixedclipping_mod] # For Differential Privacy
)


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

            # Init model
            model = SpKBGATConvOnly(
                entity_embeddings,
                relation_embeddings,
                args.entity_out_dim,
                args.entity_out_dim,
                args.drop_GAT,
                args.drop_conv,
                args.alpha,
                args.alpha_conv,
                args.nheads_GAT,
                args.out_channels,
            )
            set_parameters(model, parameters)

            path = f"{save_path}/peft_{server_round}/"
            os.makedirs(path, exist_ok=True)
            path += f"trained_{server_round}.pth"
            torch.save(model.state_dict(), path)

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
        evaluate_fn=get_evaluate_fn(args, 5, args.num_rounds, save_path),
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
