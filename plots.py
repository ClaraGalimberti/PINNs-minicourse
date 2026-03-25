import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(t_train, x_train, t_test, x_test, t_phys=None, models=None, title=""):
    plt.figure(figsize=(10, 4))
    plt.plot(t_test,  x_test,  color="tab:blue",   label="True dynamics")
    plt.plot(t_train, x_train, "x", color="tab:orange", label="Training data")
    if t_phys is not None:
        plt.plot(t_phys, np.zeros_like(t_phys), "*", ms=4,
                 color="tab:purple", label="Collocation points")
    colors = ["tab:green", "tab:pink", "tab:olive", "tab:red"]
    for (label, model), c in zip((models or {}).items(), colors):
        with torch.no_grad():
            pred = model(t_test).numpy()
        if pred.shape[-1] > 1:
            ls = ["solid", "dotted", "dashed", "dashdot"]
            for i in range(pred.shape[-1]):
                plt.plot(t_test, pred[..., i], color=c, linestyle=ls[i], label=label+"_x%i"%i)
        else:
            plt.plot(t_test, pred, color=c, label=label)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss(history: dict, title="Losses"):
    plt.figure(figsize=(8, 4))
    epochs = history["epochs"]
    for key, loss_log in ((k, v) for k, v in history.items() if k != "epochs"):
        plt.plot(epochs, loss_log, label=key)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.semilogy()
    plt.tight_layout()
    plt.show()
