import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plots import plot_predictions, plot_loss
from models import exact_solution, NeuralNet, NeuralNetWithParams
from utils import time_derivative, time_derivative2

np.random.seed(0)
torch.random.manual_seed(0)

# ODE parameters
MU = 0.4
K = 4

# Data generation parameters
N_TRAIN = 20
TMAX_TRAIN = 3
NOISE_STD = 0.1

N_TEST = 100
TMAX_TEST = 10


t_train = torch.linspace(0, TMAX_TRAIN, N_TRAIN + 1).unsqueeze(-1)
t_test = torch.linspace(0, TMAX_TEST, N_TEST + 1).unsqueeze(-1)

x_train = exact_solution(t_train, k=K, mu=MU) + NOISE_STD * torch.randn_like(t_train)
x_test = exact_solution(t_test, k=K, mu=MU)

plot_predictions(t_train, x_train, t_test, x_test, title="Data")

#%%
print("\n++++++++ Training NN model ++++++++\n")
model_nn = NeuralNet(input_dim=1, output_dim=1, hidden_dims=[32])

def train_nn(model, t_train, x_train, epochs=10000, learning_rate=1e-3, weight_decay=1e-3):
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"total": [], "epochs": []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        # compute data loss
        x_pred = model(t_train)
        loss = mse(x_pred, x_train)

        # backward pass and update parameters
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history["epochs"].append(epoch)
            history["total"].append(loss.item())
        print(f"Epoch: {epoch:5d} -- Loss: {loss.item():.4f}") if epoch % (epochs//10) == 0 else None
    return history

hist_nn = train_nn(model_nn, t_train=t_train, x_train=x_train)
plot_predictions(t_train, x_train, t_test, x_test, models={"NN": model_nn}, title="Classic NN")
plot_loss(hist_nn)

#%%
# choose points to evaluate the physics loss with
TMAX_PHYS = 7
N_PHYS = 200
t_phys = np.random.uniform(0, TMAX_PHYS, N_PHYS)

#%%
print("\n++++++++ Training PINN model ++++++++\n")
model_pinn = NeuralNet(input_dim=1, output_dim=1, hidden_dims=[32])

def train_pinn(model, t_train, x_train, t_phys, epochs=30000, learning_rate=5e-3):
    lambda_ = 1.
    mse = nn.MSELoss()
    t_phys = torch.tensor(t_phys, requires_grad=True, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "data": [], "physics": [], "epochs": []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        # compute data loss
        x_pred = model(t_train)
        loss_data = mse(x_pred, x_train)

        # compute physics loss
        x_phys = model(t_phys)
        xdot = time_derivative(x_phys, t_phys)
        xddot = time_derivative(xdot, t_phys)
        ode_residual = xddot + MU * xdot + K * x_phys
        loss_physics = lambda_ * torch.mean(ode_residual**2)

        # backward pass and update params
        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history["epochs"].append(epoch)
            history["total"].append(loss.item())
            history["data"].append(loss_data.item())
            history["physics"].append(loss_physics.item())
        if epoch % (epochs//10) == 0:
            print(f"Epoch: {epoch:5d} -- loss: {loss.item():.4f} || loss data: {loss_data.item():.4f} -- loss physics: {loss_physics.item():.4f}")
    return history

hist_pinn = train_pinn(model_pinn, t_train, x_train, t_phys)
plot_predictions(t_train, x_train, t_test, x_test, t_phys=t_phys, models={"NN": model_nn, "PINN": model_pinn}, title="PINN")
plot_loss(hist_pinn)

#%%
print("\n++++++++ Training PINN model + parameters ++++++++\n")
model_pinn_with_params = NeuralNetWithParams(input_dim=1, output_dim=1, hidden_dims=[32])

def train_pinn_with_params(model, t_train, x_train, t_phys, epochs=30000, learning_rate=5e-3):
    lambda_ = 1.
    mse = nn.MSELoss()
    t_phys = torch.tensor(t_phys, requires_grad=True, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "data": [], "physics": [], "epochs": []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        # compute data loss
        x_pred = model(t_train)
        loss_data = mse(x_pred, x_train)

        # compute physics loss (same as before with parameters mu and k added to back propagation step)
        x_phys = model(t_phys)
        xdot = time_derivative(x_phys, t_phys)
        xddot = time_derivative(xdot, t_phys)
        ode_residual = xddot + model.mu * xdot + model.k * x_phys
        loss_physics = lambda_ * torch.mean(ode_residual**2)

        # backward pass and update params
        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history["epochs"].append(epoch)
            history["total"].append(loss.item())
            history["data"].append(loss_data.item())
            history["physics"].append(loss_physics.item())
        if epoch % (epochs//10) == 0:
            print(f"Epoch: {epoch:5d} -- loss: {loss.item():.4f} || loss data: {loss_data.item():.4f} -- loss physics: {loss_physics.item():.4f}")
    return history

hist_pinn_with_params = train_pinn_with_params(model_pinn_with_params, t_train, x_train, t_phys)
print(
    f"Learned mu = {model_pinn_with_params.mu.item()}, exact mu = {MU}\nlearned k = {model_pinn_with_params.k.item()}, exact k = {K}"
)
plot_predictions(t_train, x_train, t_test, x_test, t_phys=t_phys, models={"NN": model_nn, "PINN": model_pinn, "PINN learned params": model_pinn_with_params}, title="PINN with learned params")
plot_loss(hist_pinn_with_params)


#%%
#####################################
#####################################
#####################################

print("\n++++++++ Training PINN model (state space representation) ++++++++\n")

model_pinn_2 = NeuralNet(1, 2, [16, 16])

def train_pinn_2(model, t_train, x_train, t_phys, epochs=30000, learning_rate=5e-3):
    lambda_ = 1.
    mse = nn.MSELoss()
    t_phys = torch.tensor(t_phys, requires_grad=True, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "data": [], "physics": [], "epochs": []}
    for epoch in range(epochs):
        optimizer.zero_grad()

        # compute data loss
        x_pred = model(t_train)
        x_pred_0, x_pred_1 = torch.split(x_pred, 1, dim=1)
        loss_data = mse(x_pred_0, x_train)

        # compute physics loss
        x_phys = model(t_phys)
        x_phys_0, x_phys_1 = torch.split(x_phys, 1, dim=1)
        x_deriv = time_derivative2(model, t_phys)
        xdot, xddot = torch.split(x_deriv, 1, dim=1)
        ode_residual_0 = xddot + MU * x_phys_1 + K * x_phys_0
        ode_residual_1 = xdot - x_phys_1

        loss_physics = lambda_ * (torch.mean(ode_residual_0**2) + torch.mean(ode_residual_1**2))
        loss = loss_data + loss_physics

        # backward pass and update params
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history["epochs"].append(epoch)
            history["total"].append(loss.item())
            history["data"].append(loss_data.item())
            history["physics"].append(loss_physics.item())
        # print(f"loss = {loss.item()}") if i % 500 == 0 else None
        if epoch % (epochs//10) == 0:
            print(f"Epoch: {epoch:5d} -- loss: {loss.item():.4f} || loss data: {loss_data.item():.4f} -- loss physics: {loss_physics.item():.4f}")

    return history

hist_pinn_2 = train_pinn_2(model_pinn_2, t_train, x_train, t_phys)
plot_predictions(t_train, x_train, t_test, x_test, t_phys=t_phys, models={"NN": model_nn, "PINN": model_pinn_2}, title="PINN with state space equations")
plot_loss(hist_pinn_2)

#%%
print("\n++++++++ Training PINN model (state space representation) + parameters ++++++++\n")
model_pinn_with_params_2 = NeuralNetWithParams(input_dim=1, output_dim=2, hidden_dims=[16, 16])

def train_pinn_with_params_2(model, t_train, x_train, t_phys, epochs=30000, learning_rate=5e-3):
    lambda_ = 1.
    mse = nn.MSELoss()
    t_phys = torch.tensor(t_phys, requires_grad=True, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "data": [], "physics": [], "epochs": []}
    for epoch in range(epochs):
        optimizer.zero_grad()

        # compute data loss
        x_pred = model(t_train)
        x_pred_0, x_pred_1 = torch.split(x_pred, 1, dim=1)
        loss_data = mse(x_pred_0, x_train)

        # compute physics loss
        x_phys = model(t_phys)
        x_phys_0, x_phys_1 = torch.split(x_phys, 1, dim=1)
        x_deriv = time_derivative2(model, t_phys)
        xdot, xddot = torch.split(x_deriv, 1, dim=1)
        ode_residual_0 = xddot + model.mu * x_phys_1 + model.k * x_phys_0
        ode_residual_1 = xdot - x_phys_1
        loss_physics = lambda_ * (torch.mean(ode_residual_0**2) + torch.mean(ode_residual_1**2))
        loss = loss_data + loss_physics

        # backward pass and update params
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            history["epochs"].append(epoch)
            history["total"].append(loss.item())
            history["data"].append(loss_data.item())
            history["physics"].append(loss_physics.item())
        # print(f"loss = {loss.item()}") if i % 500 == 0 else None
        if epoch % (epochs//10) == 0:
            print(f"Epoch: {epoch:5d} -- loss: {loss.item():.4f} || loss data: {loss_data.item():.4f} -- loss physics: {loss_physics.item():.4f}")

    return history

hist_pinn_with_params_2 = train_pinn_with_params_2(model_pinn_with_params_2, t_train, x_train, t_phys)
plot_predictions(t_train, x_train, t_test, x_test, t_phys=t_phys, models={"NN": model_nn, "PINN": model_pinn_with_params_2}, title="PINN with state space equations")
plot_loss(hist_pinn_with_params_2)
