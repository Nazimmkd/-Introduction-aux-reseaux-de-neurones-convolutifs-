import numpy as np


def vector_label(y, n_classes=10):
    assert np.all((y >= 0) & (y < n_classes)), f"Label hors bornes : {np.unique(y)}"
    vectors = np.zeros((n_classes, y.size))
    vectors[y, np.arange(y.size)] = 1
    return vectors


def rgb_to_grayscale(X):
    """X : (N, H, W, 3) → (N, H, W)"""
    return np.dot(X, [0.299, 0.587, 0.114])
