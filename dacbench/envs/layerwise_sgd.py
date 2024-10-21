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

def _optimizer_actions(
    optimizer: torch.optim.Optimizer, actions: list[float]
) -> None:
    for g, lr in zip(optimizer.param_groups, actions):
        g["lr"] = lr
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


class LayerwiseSGDEnv(AbstractMADACEnv):
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
        torch.use_deterministic_algorithms(True) # For reproducibility
        self.epoch_mode = config.get("epoch_mode", True)
        self.device = config.get("device")

        self.optimizer_type = torch.optim.SGD
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config["loss_function"](**config["loss_function_kwargs"])
        self.dataset_name = config.get("dataset_name")
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
        self.instance_set = config.get("instance_set")
        self.inst_id = 0

        self.predictions = deque(torch.zeros(2))

    def step(self, action: list[float]):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        truncated = super().step_()
        info = {}

        log_learning_rates = action
        self.learning_rates = [10**log_learning_rate for log_learning_rate in log_learning_rates]

        # Update action history
        self._update_lr_histories(log_learning_rates)

        self.optimizer = _optimizer_actions(
            self.optimizer, self.learning_rates
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
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss, self.train_accuracy = self.forward_backward(*train_args)

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
                self.get_state(),
                torch.tensor(self.crash_penalty),
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.c_step % len(self.train_loader) == 0 or self._done
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

        # Calculate test loss after every epoch + when done
        if self._done or self.c_step % len(self.train_loader) == 0:
            val_args = [
                self.model,
                self.loss_function,
                self.test_loader,
                self.batch_size,
                1.0,
                self.device,
            ]
            test_losses, self.test_accuracies = test(*val_args)
            self.test_loss = test_losses.mean()
            self.test_accuracy = self.test_accuracies.mean()

        reward = self.get_reward()

        return self.get_states(), reward, False, truncated, info

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        super().reset_(seed=seed)
        if options is None:
            options = {}

        # Set global seed for data loaders
        if self.instance_mode == "random_seed":
            run_seed = self.rng.integers(0, 1000000000)
        elif self.instance_mode == "instance_set":
            run_seed = self.instance[1]
        else:
            run_seed = self.initial_seed
        set_global_seeds(run_seed)

        print(f"run_seed: {run_seed}", flush=True)

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            run_seed,
            self.instance_set_path,
            self.dataset_name if self.instance_mode != "instance_set" else self.instance[0],
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
        elif self.instance_mode == "instance_set":
            self.model = sgd_utils.create_model(
                self.instance[2], len(self.datasets[0].classes)
            )
        elif self.instance_mode == "random_seed":
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )
        else:
            raise NotImplementedError(
                f"No implementation for instance version: {self.instance_mode}"
            )
        # self.lr_history = deque(torch.ones(5) * math.log10(self.initial_learning_rate))
        self.optimizer_type = torch.optim.SGD
        self.info = {}
        self._done = False

        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            **self.optimizer_params, params=self.model.parameters(), lr=self.initial_learning_rate
        )

        self.n_layers = len(self.optimizer.param_groups)
        self.learning_rates = [self.initial_learning_rate] * self.n_layers
        self.lr_histories = [deque(torch.ones(5) * math.log10(self.initial_learning_rate))] * self.n_layers
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
        test_losses, test_accuracies = test(*test_args)
        self.test_loss = test_losses.mean()
        self.test_accuracy = test_accuracies.mean()

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

        return self.get_state(), {}

    def get_default_reward(self) -> torch.tensor:
        """The default reward function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            float: The calculated reward
        """
        return torch.tensor(-self.validation_loss)

    def get_default_states(self) -> torch.Tensor:
        """Default state function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            list[dict]: The current states of all layers
        """
        # Global observations
        remaining_budget = torch.tensor([(self.n_steps - self.c_step) / self.n_steps])

        global_observations = torch.cat(
            [
                remaining_budget,
                torch.tensor([is_train_loss_finite]),
                torch.tensor([math.log10(self.initial_learning_rate)]),
                torch.tensor([self.train_loss]),
                torch.tensor([self.validation_loss]),
                torch.tensor([loss_ratio]),
                torch.tensor([self.train_accuracy.item()]),
                torch.tensor([self.validation_accuracy]),
            ]
        )

        
        # Layerspecific observations
        local_observations = []
        parameterizable_layer_idx = 0
        for layer_idx, param_group in enumerate(self.optimizer.param_groups):
            # Layer encoding
            layer = self.model[layer_idx]
            layer_type = type(layer).__name__
            

            log_learning_rate = (
                np.log10(self.learning_rates[layer_idx])
                if self.learning_rates[layer_idx] != 0
                else np.log10(1e-10)
            )
            local_observations.append(torch.tensor[log_learning_rate])

            # Accumulate all elements from weights, gradients, and velocities
            weights_all = []
            grads_all = []
            velocities_all = []

            for param in param_group["params"]:
                # Weights
                weights_all.append(param.data.view(-1))

                # Gradients
                grads_all.append(param.grad.view(-1))

                # Velocities               
                state = self.optimizer.state[param]
                velocities_all.append(state['momentum_buffer'].view(-1))

            # Concatenate all
            weights_all = torch.cat(weights_all)
            weights_mean = weights_all.mean().item()
            weights_var = weights_all.var().item()
            weights_norm = weights_all.norm(p=2).item()

            grads_all = torch.cat(grads_all)
            grads_mean = grads_all.mean().item()
            grads_var = grads_all.var().item()
            grads_norm = grads_all.norm(p=2).item()

            velocities_all = torch.cat(velocities_all)
            velocities_mean = velocities_all.mean().item()
            velocities_var = velocities_all.var().item()
            velocities_norm = velocities_all.norm(p=2).item()

            local_observations.extend(
                [
                    weights_mean,
                    weights_var,
                    weights_norm,
                    grads_mean,
                    grads_var,
                    grads_norm,
                    velocities_mean,
                    velocities_var,
                    velocities_norm,
                ]
            )

        

        optim_state = torch.cat(
            [
                remaining_budget,
                torch.tensor([log_learning_rate]),
                torch.tensor(lr_hist_deltas[1:]),  # first one always 0
                
                torch.tensor([norm_grad_layer.item()]),
                torch.tensor([norm_vel_layer.item()]),
                torch.tensor([norm_weights_layer.item()]),
                torch.tensor([mean_weight.item()]),
                torch.tensor([var_weight.item()]),
                torch.tensor([first_layer_grad_norm.item()]),
                torch.tensor([first_layer_vel_norm.item()]),
                torch.tensor([first_layer_weights_norm.item()]),
                torch.tensor([first_layer_weight_mean.item()]),
                torch.tensor([first_layer_weight_var.item()]),
                torch.tensor([last_layer_grad_norm.item()]),
                torch.tensor([last_layer_vel_norm.item()]),
                torch.tensor([last_layer_weights_norm.item()]),
                torch.tensor([last_layer_weight_mean.item()]),
                torch.tensor([last_layer_weight_var.item()]),
            ]
        )
        if self.epoch_mode:
            optim_state.append(self.average_loss)

        return optim_state

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

    def _update_lr_histories(self, log_learning_rates: list[float]) -> None:
        for lr_history, log_lr in zip(self.lr_histories, log_learning_rates):
            lr_history.pop()
            lr_history.appendleft(log_lr)

    def seed(self, seed, seed_action_space=False):
        super(LayerwiseSGDEnv, self).seed(seed, seed_action_space)

        self.rng = np.random.default_rng(seed)
