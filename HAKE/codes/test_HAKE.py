import os
import json
import logging
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader

from models import KGEModel, ModE, HAKE

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator


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
        default=128,
        help="Dropout probability for convolution layer",
    )
    parser.add_argument(
        "--sample_clients",
        type=float,
        default=4,
        help="Dropout probability for convolution layer",
    )
    parser.add_argument(
        "--num_clients",
        type=float,
        default=737,
        help="Dropout probability for convolution layer",
    )

    return parser.parse_args(args)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError("one of train/val/test mode must be choosed.")

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError("one of init_checkpoint/data_path must be choosed.")

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    data_reader = DataReader(args.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    logging.info("Model: {}".format(args.model))
    logging.info("Data Path: {}".format(args.data_path))
    logging.info("Num Entity: {}".format(num_entity))
    logging.info("Num Relation: {}".format(num_relation))

    logging.info("Num Train: {}".format(len(data_reader.train_data)))
    logging.info("Num Valid: {}".format(len(data_reader.valid_data)))
    logging.info("Num Test: {}".format(len(data_reader.test_data)))

    if args.model == "ModE":
        kge_model = ModE(num_entity, num_relation, args.hidden_dim, args.gamma)
    elif args.model == "HAKE":
        kge_model = HAKE(
            num_entity,
            num_relation,
            args.hidden_dim,
            args.gamma,
            args.modulus_weight,
            args.phase_weight,
        )

    logging.info("Model Parameter Configuration:")
    for name, param in kge_model.named_parameters():
        logging.info(
            "Parameter %s: %s, require_grad = %s"
            % (name, str(param.size()), str(param.requires_grad))
        )

    kge_model = kge_model.cuda()

    # Restore model from checkpoint directory
    logging.info("Loading checkpoint %s..." % args.init_checkpoint)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, "checkpoint"))
    init_step = 0
    kge_model.load_state_dict(checkpoint["model_state_dict"])

    step = init_step

    logging.info("Evaluating on Test Dataset...")
    metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args)
    log_metrics("Test", step, metrics)


if __name__ == "__main__":
    main(parse_args())
