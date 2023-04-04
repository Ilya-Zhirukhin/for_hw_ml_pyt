import numpy as np

def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"
    return np.mean((predictions - targets) ** 2)

def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"
    return np.mean(predictions == targets)

def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred, true in zip(predictions, targets):
        matrix[true, pred] += 1

    return matrix


