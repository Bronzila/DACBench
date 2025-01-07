"""SGD environment."""

from __future__ import annotations

import math
import os
import pickle
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from DACBench.dacbench.envs.env_utils.sgd_utils import random_torchvision_loader
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils.nanoGPT import GPT, GPTConfig


def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _optimizer_actions(optimizer: torch.optim.Optimizer, actions: list[float]) -> None:
    for g, lr in zip(optimizer.param_groups, actions, strict=True):
        g["lr"] = lr


def _get_layer_encoding(layer_type: str) -> int:
    if layer_type == "Linear":
        return 0
    if layer_type == "Conv2d":
        return 1
    raise ValueError("Unkown layer type.")


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
        torch.use_deterministic_algorithms(True)  # For reproducibility
        self.epoch_mode = config.get("epoch_mode", True)
        self.ddp_local_rank = os.environ["LOCAL_RANK"]
        self.device = config.get("device")
        self.pdtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=self.pdtype)
        )
        self.gradient_accumulation_steps = 5 * 8

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
        self.get_states = config.get("state_method", self.get_default_states)

        self.initial_learning_rate = config.get("initial_learning_rate")
        self.state_version = config.get("state_version")
        self.initial_seed = config.get("seed")
        self.seed(self.initial_seed)

        self.instance_set_path = config.get("instance_set_path")
        self.dataset_path = config.get("dataset_path")
        self.fraction_of_dataset = config.get("fraction_of_dataset")
        self.train_validation_ratio = config.get("train_validation_ratio")
        self.instance_mode = config.get("instance_mode")
        self.instance_set = config.get("instance_set")
        self.inst_id = 0

        # nanoGPT specific
        self.n_layer = config.get("n_layer")
        self.n_head = config.get("n_head")
        self.n_embd = config.get("n_embd")
        self.block_size = config.get("block_size")
        self.bias = config.get("bias")
        self.dropout = config.get("dropout")

        meta_path = Path(self.dataset_path, "meta.pkl")
        self.meta_vocab_size = None
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta["vocab_size"]

        self.predictions = deque(torch.zeros(2))

    def step(self, actions: list[float]):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        truncated = super().step_()
        info = {}

        log_learning_rates = actions
        self.learning_rates = [
            10**log_learning_rate for log_learning_rate in log_learning_rates
        ]

        # Update action history
        self._update_lr_histories(log_learning_rates)

        _optimizer_actions(self.optimizer, self.learning_rates)

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
                self.get_states(),
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

        return self.get_states(), reward, truncated, truncated, info

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        super().reset_(seed=seed, scheme="round_robin")
        if options is None:
            options = {}

        # Set global seed for data loaders
        run_seed = self.rng.integers(0, 1000000000)
        set_global_seeds(run_seed)
        print(f"run_seed: {run_seed}", flush=True)

        self.datasets, loaders = random_torchvision_loader(
            run_seed,
            self.dataset_path,
            (
                self.dataset_name
                if self.instance_mode != "instance_set"
                else self.instance[0]
            ),
            self.batch_size,
            self.fraction_of_dataset,
            self.train_validation_ratio,
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

        self.epoch_length = len(self.train_loader)

        model_args = {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "bias": self.bias,
            "vocab_size": None,
            "dropout": self.dropout,
        }  # start with model_args from command line
        # init a new model from scratch
        # determine the vocab size we'll use for from-scratch training
        # Default vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
        model_args["vocab_size"] = (
            self.meta_vocab_size if self.meta_vocab_size is not None else 50304
        )
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)

        # TODO: unsure if this is the case for us, since we don't use checkpoints
        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            model_args["block_size"] = (
                self.block_size
            )  # so that the checkpoint will have the right value
        self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(
            enabled=isinstance(self.pdtype, torch.float16)
        )

        if compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.optimizer_type = torch.optim.SGD
        self.info = {}
        self._done = False

        # create param groups then feed them into optimizer
        param_groups, self.layer_types = self._create_param_groups(
            self.optimizer_params["weight_decay"]
        )
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            param_groups,
            **self.optimizer_params,
        )

        self.n_layers = len(self.optimizer.param_groups)
        self.learning_rates = [self.initial_learning_rate] * self.n_layers
        self.lr_histories = [
            deque(torch.ones(5) * math.log10(self.initial_learning_rate))
        ] * self.n_layers

        X, Y = self.get_batch("train")  # fetch the very first batch
        raw_model = (
            self.model.module if self.ddp else self.model
        )  # unwrap DDP container if needed
        running_mfu = -1.0

        losses = self.estimate_loss()
        self.train_loss = losses["train"]
        self.val_loss = losses["val"]
        self.test_loss = losses["test"]

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_states(), {}

    def _create_param_groups(self, weight_decay) -> tuple[list[Any], list[Any]]:
        param_groups = []
        layer_types = []
        for layer in self.model.children():
            if isinstance(layer, torch.nn.Linear | torch.nn.Conv2d):
                param_groups.append(
                    {
                        "params": layer.parameters(),
                        "lr": self.initial_learning_rate,
                        "weight_decay": weight_decay if layer.dim() >= 2 else 0.0,
                    }
                )
                layer_types.append(type(layer).__name__)
        return param_groups, layer_types

    def get_default_reward(self) -> torch.tensor:
        """The default reward function.

        Returns:
            float: The calculated reward
        """
        return torch.tensor(-self.validation_loss)

    def get_default_states(self) -> list[torch.Tensor]:
        """Default state function.

        Returns:
            list[dict]: The current states of all layers
        """
        states = []
        # Global observations
        remaining_budget = torch.tensor(
            [(self.n_steps - self.c_step) / self.n_steps], device=self.device
        )
        is_train_loss_finite = int(np.isfinite(self.train_loss))
        loss_ratio = np.log(self.validation_loss / self.train_loss)

        global_observations_tensor = torch.cat(
            [
                remaining_budget,
                torch.tensor([is_train_loss_finite], device=self.device),
                torch.tensor(
                    [math.log10(self.initial_learning_rate)], device=self.device
                ),
                torch.tensor([self.train_loss], device=self.device),
                torch.tensor([self.validation_loss], device=self.device),
                torch.tensor([loss_ratio], device=self.device),
                torch.tensor([self.train_accuracy.item()], device=self.device),
                torch.tensor([self.validation_accuracy], device=self.device),
            ]
        )

        # Layerspecific observations
        for layer_idx, param_group in enumerate(self.optimizer.param_groups):
            local_observations = []
            # Layer encoding
            layer_type = self.layer_types[layer_idx]
            layer_enc = _get_layer_encoding(layer_type)
            local_observations.append(torch.tensor([layer_enc], device=self.device))

            log_learning_rate = (
                np.log10(self.learning_rates[layer_idx])
                if self.learning_rates[layer_idx] != 0
                else np.log10(1e-10)
            )
            local_observations.append(
                torch.tensor([log_learning_rate], device=self.device)
            )
            lr_hist_deltas = log_learning_rate - self.lr_histories[layer_idx]
            local_observations.append(torch.tensor(lr_hist_deltas, device=self.device))

            depth_enc = layer_idx / len(self.optimizer.param_groups)
            local_observations.append(torch.tensor([depth_enc], device=self.device))

            # Weight, gradient and momentum statistics
            weights_all = []
            grads_all = []
            velocities_all = []

            for param in param_group["params"]:
                # Weights
                weights = param.data.view(-1)
                weights_all.append(weights)

                # Gradients
                if param.grad is not None:
                    grads_all.append(param.grad.view(-1))
                else:
                    assert self.c_step < 2
                    grads_all.append(torch.zeros_like(weights))

                # Velocities
                state = self.optimizer.state[param]
                if "momentum_buffer" in state:
                    velocities_all.append(state["momentum_buffer"].view(-1))
                else:
                    assert self.c_step < 2
                    velocities_all.append(torch.zeros_like(weights))

            # Concatenate all
            weights_tensor = torch.cat(weights_all)
            weights_mean = weights_tensor.mean().unsqueeze(0).to(self.device)
            weights_var = weights_tensor.var().unsqueeze(0).to(self.device)
            weights_norm = weights_tensor.norm(p=2).unsqueeze(0).to(self.device)

            grads_tensor = torch.cat(grads_all)
            grads_mean = grads_tensor.mean().unsqueeze(0).to(self.device)
            grads_var = grads_tensor.var().unsqueeze(0).to(self.device)
            grads_norm = grads_tensor.norm(p=2).unsqueeze(0).to(self.device)

            velocities_tensor = torch.cat(velocities_all)
            velocities_mean = velocities_tensor.mean().unsqueeze(0).to(self.device)
            velocities_var = velocities_tensor.var().unsqueeze(0).to(self.device)
            velocities_norm = velocities_tensor.norm(p=2).unsqueeze(0).to(self.device)

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

            local_observations_tensor = torch.cat(local_observations)
            layer_state = torch.cat(
                [local_observations_tensor, global_observations_tensor]
            )
            states.append(layer_state)

        return states

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

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate an arbitrarily accurate loss over either split using many batches."""
        out = {}
        self.model.eval()
        for loader, name in zip(
            [self.train_loader, self.validation_loader, self.test_loader],
            ["train", "val", "test"],
            strict=False,
        ):
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = next(iter(loader))
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[name] = losses.mean()
        self.model.train()
        return out

    def run_epoch(self):
        """Run a single epoch of training for given `model` with `loss_function`."""

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(self.gradient_accumulation_steps):
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
            with self.ctx:
                logits, loss = self.model(X, Y)
                loss = loss / self.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = self.train_loader.
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
        
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
        for lr_history, log_lr in zip(
            self.lr_histories, log_learning_rates, strict=True
        ):
            lr_history.pop()
            lr_history.appendleft(log_lr)

    def seed(self, seed, seed_action_space=False):
        super().seed(seed, seed_action_space)

        self.rng = np.random.default_rng(seed)


### Do we need to implement this scheduler or use our predefined ones?

# learning rate decay scheduler (cosine with warmup)
# def get_lr(it):
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_iters:
#         return learning_rate * it / warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > lr_decay_iters:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
#     return min_lr + coeff * (learning_rate - min_lr)
