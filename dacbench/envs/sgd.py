"""SGD environment."""

from __future__ import annotations
import math
import random

import numpy as np
import torch

from collections import deque
from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils import sgd_utils
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader


def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class SGD_Momentum(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if dampening != 0.0:
            raise NotImplementedError("Dampening not implemented.")
        if weight_decay != 0:
            raise NotImplementedError("Weight decay not implemented.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super(SGD_Momentum, self).__init__(params, defaults)
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = dict(
                    vel=torch.zeros_like(p.data),
                    grad=torch.zeros_like(p.data),
                    weights=torch.zeros_like(p.data),
                )

    def step(
        self,
    ):
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    self.state[p] = dict(
                        vel=torch.zeros_like(p.data),
                        grad=torch.zeros_like(p.data),
                        weights=torch.zeros_like(p.data),
                    )

                self.state[p]["grad"] = p.grad.data
                self.state[p]["weights"] = p.data
                velocity = self.state[p]["vel"]
                velocity.mul_(self.momentum)
                velocity.add_(group["lr"] * p.grad.data)

                p.data -= velocity


def _optimizer_action(
    optimizer: torch.optim.Optimizer, action: float, use_momentum: bool
) -> None:
    for g in optimizer.param_groups:
        g["lr"] = action
        # if use_momentum:
        #     print("Momentum")
        #     g["betas"] = (action[1], 0.999)
    return optimizer


def test(
    model,
    loss_function,
    loader,
    batch_size,
    batch_percentage: float = 1.0,
    device="cpu",
):
    """Evaluate given `model` on `loss_function`.

    Percentage defines how much percentage of the data shall be used.
    If nothing given the whole data is used.

    Returns:
        test_losses: Batch validation loss per data point
    """
    nmb_sets = batch_percentage * (len(loader.dataset) / batch_size)
    model.eval()
    test_losses = []
    test_accuracies = []
    i = 0

    with torch.no_grad():
        for data, target in loader:
            d_data, d_target = data.to(device), target.to(device)
            output = model(d_data)
            _, preds = output.max(dim=1)
            test_losses.append(loss_function(output, d_target))
            test_accuracies.append(torch.sum(preds == d_target) / len(d_target))
            i += 1
            if i >= nmb_sets:
                break
    return (
        torch.cat(test_losses).cpu().numpy(),
        torch.tensor(test_accuracies).cpu().numpy(),
    )


class SGDEnv(AbstractMADACEnv):
    """The SGD DAC Environment implements the problem of dynamically configuring
    the learning rate hyperparameter of a neural network optimizer
    (more specifically, torch.optim.AdamW) for a supervised learning task.
    While training, the model is evaluated after every epoch.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of model on test_loader of the instance       if done
        crash_penalty of the instance                               if crashed
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, config):
        """Init env."""
        super().__init__(config)
        self.epoch_mode = config.get("epoch_mode", True)
        self.device = config.get("device")

        self.optimizer_type = SGD_Momentum
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config["loss_function"](**config["loss_function_kwargs"])
        self.dataset_name = config.get("dataset_name")
        self.use_momentum = config.get("use_momentum")
        self.use_generator = config.get("model_from_dataset")
        self.torchub_model = config.get("torch_hub_model", (False, None, False))

        # Use default reward function, if no specific function is given
        self.get_reward = config.get("reward_function", self.get_default_reward)

        # Use default state function, if no specific function is given
        self.get_state = config.get("state_method", self.get_default_state)

        self.learning_rate = config.get("initial_learning_rate")
        self.initial_learning_rate = config.get("initial_learning_rate")
        self.state_version = config.get("state_version")
        self.initial_seed = config.get("seed")
        self.seed(self.initial_seed)

        self.instance_set_path = config.get("instance_set_path")
        self.fraction_of_dataset = config.get("fraction_of_dataset")
        self.train_validation_ratio = config.get("train_validation_ratio")
        self.instance_mode = config.get("instance_mode")
        self.inst_id = 0

        self.lr_history = deque(torch.ones(5) * math.log10(self.initial_learning_rate))
        self.predictions = deque(torch.zeros(2))

    def step(self, action: float):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        truncated = super().step_()
        info = {}

        log_learning_rate = action
        self.learning_rate = 10**log_learning_rate

        # Update action history
        self.lr_history.pop()
        self.lr_history.appendleft(log_learning_rate)

        self.optimizer = _optimizer_action(
            self.optimizer, self.learning_rate, self.use_momentum
        )

        if self.epoch_mode:
            self.train_loss, self.average_loss = self.run_epoch(
                self.model,
                self.loss_function,
                self.train_loader,
                self.optimizer,
                self.device,
            )
        else:
            train_args = [
                self.model,
                self.loss_function,
                self.train_loader,
                self.device,
            ]
            self.optimizer.zero_grad()
            self.train_loss, self.train_accuracy = self.forward_backward(*train_args)
            self.optimizer.step()

        crashed = (
            not np.isfinite(self.train_loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        self.train_loss = self.train_loss.item()

        if crashed:
            self._done = True
            return (
                self.get_state(self),
                torch.tensor(self.crash_penalty),
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.n_steps % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            batch_percentage = 1.0
        else:
            batch_percentage = 0.1

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            batch_percentage,
            self.device,
        ]
        validation_loss, validation_accuracy = test(*val_args)

        self.validation_loss = validation_loss.mean()
        self.validation_accuracy = validation_accuracy.mean()
        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        if self._done:
            val_args = [
                self.model,
                self.loss_function,
                self.test_loader,
                self.batch_size,
                1.0,
                self.device,
            ]
            self.test_losses, self.test_accuracies = test(*val_args)

        reward = self.get_reward(self)

        return self.get_state(self), reward, False, truncated, info

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        if options is None:
            options = {}

        # Set global seed for data loaders
        if self.instance_mode == "random_seed":
            run_seed = self.rng.integers(0, 1000000000)
        else:
            run_seed = self.initial_seed
        set_global_seeds(run_seed)

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            run_seed,
            self.instance_set_path,
            self.dataset_name,
            self.batch_size,
            self.fraction_of_dataset,
            self.train_validation_ratio,
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

        self.epoch_length = len(self.train_loader)

        if self.instance_mode == "random_instances":
            (
                self.model,
                self.optimizer_params,
                self.batch_size,
                self.crash_penalty,
            ) = sgd_utils.random_instance(self.rng, self.datasets)
        elif self.instance_mode == "instance_sets":
            self.model = sgd_utils.create_model(
                self.config.instance_sets[self.inst_id], len(self.datasets[0].classes)
            )
            self.inst_id += 1
        elif self.instance_mode == "random_seed":
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )
        else:
            raise NotImplementedError(
                f"No implementation for instance version: {self.instance_mode}"
            )

        self.learning_rate = self.initial_learning_rate
        self.optimizer_type = SGD_Momentum
        self.info = {}
        self._done = False

        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            **self.optimizer_params, params=self.model.parameters()
        )

        # Evaluate model initially
        train_args = [
            self.model,
            self.loss_function,
            self.train_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        losses, train_accuracy = test(*train_args)
        self.train_loss = losses.mean()
        self.train_accuracy = train_accuracy.mean()

        test_args = [
            self.model,
            self.loss_function,
            self.test_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        self.test_losses, test_accuracies = test(*test_args)

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        validation_loss, validation_accuracy = test(*val_args)

        self.validation_loss = validation_loss.mean()
        self.validation_accuracy = validation_accuracy.mean()

        self.min_validation_loss = None

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_state(self), {}

    def get_default_reward(self, _) -> torch.tensor:
        """The default reward function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            float: The calculated reward
        """
        if self.test_losses is not None:
            reward = self.test_losses.sum().item() / len(self.test_loader.dataset)
        else:
            reward = 0.0
        return torch.tensor(-reward)

    def get_default_state(self, _) -> torch.Tensor:
        """Default state function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            dict: The current state
        """

        remaining_budget = torch.tensor(self.c_step - self.n_steps)
        log_learning_rate = (
            torch.log10(self.learning_rate)
            if self.learning_rate != 0
            else math.log10(1e-10)
        )
        lr_hist_deltas = self.lr_history - log_learning_rate
        prediction_variance = self.predictions[0].var()
        prediction_change_variance = (self.predictions[1] - self.predictions[0]).var()

        optimizer_state = self.optimizer.state_dict()
        norm_grad_layer = torch.ones(len(optimizer_state))
        norm_vel_layer = torch.ones(len(optimizer_state))
        norm_data_layer = torch.ones(len(optimizer_state))
        for i, param in enumerate(optimizer_state.keys()):
            norm_grad_layer[i] = optimizer_state[param]["grad"].norm(p=2)
            norm_vel_layer[i] = optimizer_state[param]["vel"].norm(p=2)
            norm_data_layer[i] = optimizer_state[param]["data"].norm(p=2)


        state = torch.cat([
            remaining_budget,
            log_learning_rate,
            lr_hist_deltas[1:], # first one always 0
            self.train_loss,
            self.validation_loss,
            self.train_accuracy,
            self.validation_accuracy,
            prediction_variance,
            prediction_change_variance,
            norm_grad_layer,
            norm_vel_layer,
            norm_data_layer,
            self._done,
        ])
        if self.epoch_mode:
            state.append(self.average_loss)

        return state

    def render(self, mode="human"):
        """Render progress."""
        if mode == "human":
            epoch = 1 + self.c_step // len(self.train_loader)
            epoch_cutoff = self.n_steps // len(self.train_loader)
            batch = 1 + self.c_step % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "  # noqa: E501
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.train_loss}, "
                f"val_loss {self.validation_loss}"
            )
        else:
            raise NotImplementedError

    def forward_backward(self, model, loss_function, loader, device="cpu"):
        """Do a forward and a backward pass for given `model` for `loss_function`.

        Returns:
            loss: Mini batch training loss per data point
        """
        model.train()
        (data, target) = next(iter(loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        loss.mean().backward()

        # for comparing current and last predictions
        self.predictions.pop()
        self.predictions.appendleft(output)

        accuracy = torch.sum(output.argmax(dim=1) == target) / len(target)
        return loss.mean().detach().cpu().numpy(), torch.tensor(accuracy).cpu().numpy()

    def run_epoch(self, model, loss_function, loader, optimizer, device="cpu"):
        """Run a single epoch of training for given `model` with `loss_function`."""
        last_loss = None
        running_loss = 0
        predictions = []
        model.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.mean().backward()
            optimizer.step()
            predictions.append(output)
            last_loss = loss
            running_loss += last_loss.mean().item()
        self.predictions.pop()
        self.predictions.appendleft(predictions.mean())
        return last_loss.mean().detach().cpu(), running_loss.cpu() / len(loader)

    def seed(self, seed, seed_action_space=False):
        super(SGDEnv, self).seed(seed, seed_action_space)

        self.rng = np.random.default_rng(seed)
