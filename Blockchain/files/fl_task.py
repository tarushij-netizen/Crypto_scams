"""
fl_task.py
The actual federated learning task: clients hold private synthetic
data, all trying to jointly learn a shared linear model
    y = X . w_true + noise
without sharing raw data -- only (masked) gradients are submitted.
"""

import numpy as np

DIM = 5
TRUE_W = np.array([1.5, -2.0, 0.5, 3.0, -1.0])


def generate_client_data(seed, num_samples=60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, DIM))
    y = X @ TRUE_W + rng.normal(scale=0.1, size=num_samples)
    return X, y


def local_gradient(X, y, w):
    """Mean-squared-error gradient w.r.t. w for this client's local data."""
    preds = X @ w
    error = preds - y
    grad = X.T @ error / len(y)
    return grad


def mse(X, y, w):
    preds = X @ w
    return float(np.mean((preds - y) ** 2))
