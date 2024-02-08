from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import pandas as pd
from dacbench.envs.env_utils.function_definitions import Rosenbrock, Rastrigin, Ackley, Sphere

from dacbench import AbstractEnv

class ToySGD2DEnv(AbstractEnv):
    """
    Optimize toy functions with SGD + Momentum.

    Action: [log_learning_rate, log_momentum] (log base 10)
    State: Dict with entries remaining_budget, gradient, learning_rate, momentum
    Reward: negative log regret of current and true function value

    An instance can look as follows:
    ID                                                  0
    function                                   Rosenbrock
    low                                                -5
    high                                                5

    """

    def __init__(self, config):
        """Init env."""
        super(ToySGD2DEnv, self).__init__(config)
        self.velocity = 0
        self.dimensions = 2
        self.gradient = torch.zeros(self.dimensions)
        self.history = []
        self.problem = None
        self.objective_function = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = config["initial_momentum"]
        self.initial_momentum = config["initial_momentum"]
        self.learning_rate = config["initial_learning_rate"]
        self.initial_learning_rate = config["initial_learning_rate"]
        self.lower_bound = config["low"]
        self.upper_bound = config["high"]
        self.function = config["function"]

    def build_objective_function(self):
        """Make base function."""
        if self.function == "Rosenbrock":
            self.problem = Rosenbrock()
        elif self.function == "Rastrigin":
            self.problem = Rastrigin()
        elif self.function == "Ackley":
            self.problem = Ackley()
        elif self.function == "Sphere":
            self.problem = Sphere()
        else:
            raise NotImplementedError(
                "Function not found."
            )
        self.x_min = self.problem.x_min
        self.f_min = self.problem.f_min
        self.objective_function = self.problem.objective_function

        self.x_cur = torch.FloatTensor(2).uniform_(self.lower_bound, self.upper_bound)

    def clip_gradient(self):
        self.gradient = torch.clip(self.gradient, -100, 100)

    def step(
        self, action: float
    ) -> Tuple[Dict[str, float], float, bool, Dict]:
        """
        Take one step with SGD.

        Parameters
        ----------
        action: float
            log_learning_rate

        Returns
        -------
        Tuple[torch.tensor, float, bool, Dict]

            - state : torch.tensor
            - reward : float
            - terminated : bool
            - truncated : bool
            - info : Dict

        """
        truncated = super(ToySGD2DEnv, self).step_()
        info = {}

        # parse action
        
        log_learning_rate = action
        self.learning_rate = 10**log_learning_rate

        # SGD + Momentum update
        self.velocity = (
            self.momentum * self.velocity + self.learning_rate * self.gradient
        )
        self.x_cur -= self.velocity

        # Clip position to optimization bounds
        self.x_cur = torch.clip(self.x_cur, self.lower_bound, self.upper_bound)
        
        # Reward
        # current function value
        x_cur_tensor = torch.tensor(self.x_cur, requires_grad=True)
        self.f_cur = self.objective_function(x_cur_tensor)
        # Gradient
        self.f_cur.backward()
        self.gradient = x_cur_tensor.grad
        self.clip_gradient()
        # log regret
        log_regret = torch.log10(torch.abs(self.f_min - self.f_cur))
        reward = -log_regret

        # State
        remaining_budget = self.n_steps - self.c_step
        
        state = torch.tensor([remaining_budget, self.learning_rate, self.gradient[0], self.gradient[1]])

        self.history.append(self.x_cur)

        return state, reward.detach(), False, truncated, info

    def reset(self, seed=None, options={}):
        """
        Reset environment.

        Parameters
        ----------
        seed : int
            seed
        options : dict
            options dict (not used)

        Returns
        -------
        torch.Tensor
            Environment state
        dict
            Meta-info

        """
        super(ToySGD2DEnv, self).reset_(seed)

        self.velocity = 0
        self.gradient = torch.zeros(self.dimensions)
        self.history = []
        self.objective_function = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.problem = None
        self.momentum = self.initial_momentum
        self.learning_rate = self.initial_learning_rate
        self.n_steps = 0
        self.build_objective_function()
        remaining_budget = self.n_steps - self.c_step
        return torch.tensor([remaining_budget, self.learning_rate, self.gradient[0], self.gradient[1]]), {"start": self.x_cur.tolist()}

    def render(self, **kwargs):
        """Render progress."""
        import matplotlib.pyplot as plt

        history = np.array(self.history).flatten()
        X = np.linspace(1.05 * np.amin(history), 1.05 * np.amax(history), 100)
        Y = self.objective_function(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X, Y, label="True")
        ax.plot(
            history,
            self.objective_function(history),
            marker="x",
            color="black",
            label="Observed",
        )
        ax.plot(
            self.x_cur,
            self.objective_function(self.x_cur),
            marker="x",
            color="red",
            label="Current Optimum",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")        
        plt.show()

    def close(self):
        """Close env."""
        pass
