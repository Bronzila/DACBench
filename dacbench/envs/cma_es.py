"""CMA ES Environment."""
from __future__ import annotations

import re

import numpy as np
from IOHexperimenter import IOH_function
from modcma import ModularCMAES, Parameters
import torch

from dacbench import AbstractMADACEnv


class CMAESEnv(AbstractMADACEnv):
    """The CMA ES environment controlles the step size on BBOB functions."""

    def __init__(self, config):
        """Initialize the environment."""
        super().__init__(config)

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if options is None:
            options = {}
        super().reset_(seed)
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.objective = IOH_function(
            self.fid, self.dim, self.iid, target_precision=1e-8
        )
        self.es = ModularCMAES(
            self.objective,
            parameters=Parameters.from_config_array(
                self.dim, np.array(self.representation).astype(int)
            ),
        )
        return self.get_state(self), {}

    def step(self, action):
        """Make one step of the environment."""
        truncated = super().step_()
        
        self.es.parameters.sigma = action

        terminated = not self.es.step()
        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def close(self):
        """Closes the environment."""
        return True

    def get_default_reward(self, *_):
        """The default reward function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            float: The calculated reward
        """
        return max(
            self.reward_range[0], min(self.reward_range[1], -self.es.parameters.fopt)
        )

    def get_default_state(self, *_):
        """Default state function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            dict: The current state
        """
        return  torch.tensor(np.array(
            [
                self.es.parameters.lambda_,
                self.es.parameters.sigma,
                self.budget - self.es.parameters.used_budget,
                self.fid,
                self.iid,
            ]
        ))

    def render(self, mode="human"):
        """Render progress."""
        raise NotImplementedError("CMA-ES does not support rendering at this point")
