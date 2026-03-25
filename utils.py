import torch
from torch.func import jacrev, vmap


def time_derivative(x, t):
    """Returns the time derivative of x at times t using automatic differentiation.
    Example:
        t = torch.linspace(0, 1, 10)
        x = model(t)
        xdot = time_derivative(x, t)"""
    xdot = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    return xdot

def time_derivative2(model, t):
    """
    Computes dx/dt for x = model(t)
    t: [N,1]
    returns: [N,2]
    """
    def f(t_single):
        return model(t_single.unsqueeze(0)).squeeze(0)
    jac = vmap(jacrev(f))(t)
    return jac.squeeze(-1)
