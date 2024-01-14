import torch
import numpy as np

class Rosenbrock():
    def __init__(self) -> None:
        self.x_min = torch.tensor([1, 1])
        self.f_min = 0

    def objective_function(self, input):
        x = input[0]
        y = input[1]
        return torch.pow((1 - x), 2) + 100 * torch.pow((y - x ** 2), 2)

class Rastrigin():
    def __init__(self) -> None:
        self.x_min = torch.tensor([0, 0])
        self.f_min = 0

    def objective_function(self, input):
        x = input[0]
        y = input[1]
        return 20 + torch.pow(x, 2) - 10 * torch.cos(2 * np.pi * x) + torch.pow(y, 2) - 10 * torch.cos(2 * np.pi * y)