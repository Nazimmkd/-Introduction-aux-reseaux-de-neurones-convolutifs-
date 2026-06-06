try:
    import cupy as xp
    _t = xp.ones((2, 2))
    xp.dot(_t, _t)
except Exception:
    import numpy as xp


def matrices(n_x, n_y):
    A = xp.random.randn(n_y, n_x) * 0.01
    b = xp.zeros((n_y, 1))
    return A, b

def fonction_score(X, A, b):
    return xp.dot(A, xp.asarray(X)) + b

def softmax(Z):
    exp_Z = xp.exp(Z - xp.max(Z, axis=0, keepdims=True))
    return exp_Z / xp.sum(exp_Z, axis=0, keepdims=True)

def log_loss(Y, P):
    n = Y.shape[1]
    return -1 / n * xp.sum(xp.asarray(Y) * xp.log(P + 1e-15))

def gradient(X, Y, P):
    n = X.shape[1]
    X = xp.asarray(X)
    Y = xp.asarray(Y)
    dA = (1 / n) * xp.dot(P - Y, X.T)
    db = (1 / n) * xp.sum(P - Y, axis=1, keepdims=True)
    return dA, db
