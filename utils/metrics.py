import numpy as np

def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculates the mean squared error between predictions and targets."""
    return np.mean((predictions - targets) ** 2)

