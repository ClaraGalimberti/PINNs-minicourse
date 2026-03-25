import numpy as np
import torch
import torch.nn as nn


def exact_solution(t, k, mu):
    """Get exact solution to the 1D under-damped harmonic oscillator."""
    assert mu**2 < 4 * k, "System must be under-damped."
    w = np.sqrt(4 * k - mu**2) / 2
    x = torch.exp(-mu / 2 * t) * torch.cos(w * t)
    return x

class NeuralNet(nn.Module):
    """Defines a pytorch neural network with one hidden layer."""

    def __init__(self, input_dim=1, output_dim=1, hidden_dims=[32]):
        super().__init__()
        assert len(hidden_dims) > 0
        modules = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh()
        ]
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.LayerNorm(hidden_dims[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


class NeuralNetWithParams(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.k  = nn.Parameter(torch.tensor(0.0))
