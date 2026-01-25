import math

import torch
from torch import nn


def stochastic_gradient_descent(params, lr=0.01, momentum=0.9):
    for p in params:



def loss_function(expected: int, actual: int):
    return math.sqrt((expected - actual) ** 2) / (expected - actual)


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = nn.ParameterList()


    def forward(self, x, lr=-0.01):
        return self.params[0](x) - (lr * )

