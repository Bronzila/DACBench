"""Benchmark for SGD."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import LayerwiseSGDEnv
from dacbench.envs.env_utils import sgd_utils

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CS.Float(name="learning_rate", bounds=(0.0, 0.05))
DEFAULT_CFG_SPACE.add_hyperparameter(LR)


def __default_loss_function(**kwargs):
    return nn.NLLLoss(reduction="none", **kwargs)


INFO = {
    "identifier": "LR",
    "name": "Layerwise Learning Rate Adaption for Neural Networks",
    "reward": "Negative Log Differential Validation Loss",
    "state_description": [
        "Step",
        "Current Learning Rate",
        "Loss",
        "Validation Loss",
        "Crashed",
        # "Predictive Change Variance (Discounted Average)",
        # "Predictive Change Variance (Uncertainty)",
        # "Loss Variance (Discounted Average)",
        # "Loss Variance (Uncertainty)",
        # "Training Loss",
        # "Alignment",
    ],
    "action_description": ["Learning Rate"],
}


LAYERWISE_SGD_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "step": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "learning_rate": spaces.Box(low=0, high=1, shape=(1,)),
                "loss": spaces.Box(0, np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "crashed": spaces.Discrete(1),
            }
        ],
        "reward_range": [-(10**9), (10**9)],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_from_dataset": False,  # If true, generates:
        # random model, optimizer_params, batch_size, crash_penalty
        "layer_specification": [
            (
                sgd_utils.LayerType.CONV2D,
                {"in_channels": 1, "out_channels": 32, "kernel_size": 3},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (
                sgd_utils.LayerType.CONV2D,
                {"in_channels": 32, "out_channels": 64, "kernel_size": 3},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (sgd_utils.LayerType.POOLING, {"kernel_size": 2}),
            (sgd_utils.LayerType.DROPOUT, {"p": 0.25}),
            (sgd_utils.LayerType.FLATTEN, {"start_dim": 1}),
            (
                sgd_utils.LayerType.LINEAR,
                {"in_features": 9216, "out_features": 128},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (sgd_utils.LayerType.DROPOUT, {"p": 0.25}),
            (sgd_utils.LayerType.LINEAR, {"in_features": 128, "out_features": 10}),
            (sgd_utils.LayerType.LOGSOFTMAX, {"dim": 1}),
        ],
        "torch_hub_model": (False, False, False),
        "optimizer_params": {
            # "weight_decay": 10.978902603194243,
            # "eps": 1.2346464628039852e-10,
            # "betas": (0.9994264825468422, 0.9866804882743139),
            "momentum": 0.9,
            "lr": 0.002,
        },
        "cutoff": 1e2,
        "loss_function": __default_loss_function,
        "loss_function_kwargs": {},
        "training_batch_size": 64,
        "fraction_of_dataset": 0.6,
        "train_validation_ratio": 0.8,  # If set to None, random value is used
        "dataset_name": "MNIST",  # If set to None, random data set is chosen;
        # else specific set can be set: e.g. "MNIST"
        # "reward_function":,    # Can be set, to replace the default function
        # "state_method":,       # Can be set, to replace the default function
        "seed": 0,
        "crash_penalty": -10000.0,
        "initial_learning_rate": 0.002,
        "multi_agent": False,
        "instance_set_path": "../instance_sets/sgd/sgd_train_100instances.csv",
        "benchmark_info": INFO,
        "dataset": "MNIST",  # MNIST, CIFAR10, FashionMNIST
        "epoch_mode": False,
        "instance_mode": "random_seed",
    }
)


class LayerwiseSGDBenchmark(AbstractBenchmark):
    """Benchmark with default configuration & relevant functions for SGD."""

    def __init__(self, config_path=None, config=None):
        """Initialize layerwise SGD Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(LAYERWISE_SGD_DEFAULTS.copy())

        for key in LAYERWISE_SGD_DEFAULTS:
            if (key not in self.config or
                key == "instance_mode" and self.config[key] == ""):
                self.config[key] = LAYERWISE_SGD_DEFAULTS[key]

    def get_environment(self):
        """Return LayerwiseSGDEnv env with current configuration.

        Returns:
        -------
        LayerwiseSGDEnv
            SGD environment
        """
        if "instance_set" not in self.config:
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config and "test_set_path" in self.config:
            self.read_instance_set(test=True)

        env = LayerwiseSGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        print(f"Running on device: {self.config['device']}")

        return env

    def read_instance_set(self, test=False):
        """Read path of instances from config into list."""
        if test:
            path = Path(__file__).resolve().parent / self.config.test_set_path
            keyword = "test_set"
        else:
            path = Path(__file__).resolve().parent / self.config["instance_set_path"]
            keyword = "instance_set"
        self.config[keyword] = {}
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                if "_" in row["dataset"]:
                    dataset_info = row["dataset"].split("_")
                    dataset_name = dataset_info[0]
                    dataset_size = int(dataset_info[1])
                else:
                    dataset_name = row["dataset"]
                    dataset_size = None

                architecture = []
                for layer in row["architecture"].split("-"):
                    match = re.match(r"(\w+)\((.*)\)", layer)
                    if match:
                        # Extract the literals and the values inside the brackets
                        layer_type = match.group(1)
                        params = match.group(2).split(",")
                        # Convert the values to integers if necessary
                        layer_params = [int(p.strip()) for p in params]
                        architecture.append((layer_type, layer_params))
                    else:
                        architecture.append((layer, []))
                instance = [
                    dataset_name,
                    int(row["seed"]),
                    architecture,
                    int(row["steps"]),
                    dataset_size,
                ]
                self.config[keyword][int(row["ID"])] = instance

    def get_benchmark(self, instance_set_path=None):
        """Get benchmark from the LTO paper.

        Parameters
        -------
        seed : int
            Environment seed

        Returns:
        -------
        env : LayerwiseSGDEnv
            SGD environment
        """
        if instance_set_path is not None:
            self.config["instance_set_path"] = instance_set_path
        self.read_instance_set()
        return LayerwiseSGDEnv(self.config)
