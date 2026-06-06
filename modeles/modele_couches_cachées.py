try:
    import cupy as xp
    _t = xp.ones((2, 2))
    xp.dot(_t, _t)
except Exception:
    import numpy as xp


# ── Fonctions d'activation ────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + xp.exp(-x))

def tanh(x):
    return xp.tanh(x)

def relu(x):
    return xp.maximum(0, x)

def heaviside(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = xp.exp(x - xp.max(x, axis=0, keepdims=True))
    return exp_x / xp.sum(exp_x, axis=0, keepdims=True)

def derivee_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def _activation(Z, nom):
    if nom == "sigmoid":   return sigmoid(Z)
    if nom == "tanh":      return tanh(Z)
    if nom == "relu":      return relu(Z)
    if nom == "heaviside": return heaviside(Z)
    raise ValueError(f"Activation inconnue : '{nom}'")


def _derivee(Z, nom):
    if nom == "sigmoid":
        s = sigmoid(Z)
        return s * (1 - s)
    if nom == "tanh":
        return 1 - xp.tanh(Z) ** 2
    if nom == "relu":
        return (Z > 0).astype(float)
    if nom == "heaviside":
        # Dérivée nulle → les gradients ne remontent pas → le modèle n'apprend pas
        return xp.zeros_like(Z)
    raise ValueError(f"Activation inconnue : '{nom}'")


# ── Initialisation ────────────────────────────────────────────

def _echelle_init(n_entree, activation):
    """
    Choisit l'échelle d'initialisation selon l'activation :
      sigmoid / tanh → Xavier  √(1/n)   — préserve la variance en entrée
      relu           → He      √(2/n)   — compense les neurones mis à zéro par ReLU
      heaviside      → 0.01              — n'a pas d'importance (ne converge pas)
    """
    if activation == "relu":
        return xp.sqrt(2.0 / n_entree)
    if activation == "heaviside":
        return 0.01
    return xp.sqrt(1.0 / n_entree)


def initialisation_mlp(couches, activation="relu"):
    """
    Initialise les paramètres d'un MLP de profondeur arbitraire.

    couches : liste complète [n_entrée, n_h1, n_h2, ..., n_sortie]
              Ex : [784, 128, 64, 10]  →  2 couches cachées

    activation : "sigmoid" | "tanh" | "relu" | "heaviside"
                 Détermine l'échelle d'initialisation de Xavier / He.
    """
    params = {}
    for l in range(1, len(couches)):
        echelle = _echelle_init(couches[l - 1], activation)
        params[f"W{l}"] = xp.random.randn(couches[l], couches[l - 1]) * echelle
        params[f"b{l}"] = xp.zeros((couches[l], 1))
    return params


# ── Propagation avant ─────────────────────────────────────────

def forward_pass(X, params, activation="relu"):
    """
    Propagation avant pour un MLP de profondeur arbitraire.

    Couches cachées  : activation choisie
    Couche de sortie : toujours softmax

    Cache retourné : {"A0": X, "Z1": ..., "A1": ..., ..., "ZL": ..., "AL": softmax}
    La sortie finale est toujours cache["A{L}"] où L = len(params)//2.
    """
    X = xp.asarray(X)
    L = len(params) // 2     # nombre total de couches (cachées + sortie)
    cache = {"A0": X}

    for l in range(1, L):    # couches cachées
        Z = xp.dot(params[f"W{l}"], cache[f"A{l - 1}"]) + params[f"b{l}"]
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = _activation(Z, activation)

    # Couche de sortie — toujours softmax (classification multiclasse)
    Z = xp.dot(params[f"W{L}"], cache[f"A{L - 1}"]) + params[f"b{L}"]
    cache[f"Z{L}"] = Z
    cache[f"A{L}"] = softmax(Z)

    return cache


# ── Rétropropagation ──────────────────────────────────────────

def backpropagation(X, Y, cache, params, activation="relu"):
    """
    Rétropropagation pour un MLP de profondeur arbitraire.

    La dérivée combinée softmax + cross-entropy simplifie à : dZ_sortie = A_L - Y
    Retourne : {"dW1": ..., "db1": ..., ..., "dWL": ..., "dbL": ...}
    """
    X = xp.asarray(X)
    Y = xp.asarray(Y)
    n = X.shape[1]
    L = len(params) // 2
    grads = {}

    # Gradient couche de sortie
    dZ = cache[f"A{L}"] - Y
    grads[f"dW{L}"] = (1 / n) * xp.dot(dZ, cache[f"A{L - 1}"].T)
    grads[f"db{L}"] = (1 / n) * xp.sum(dZ, axis=1, keepdims=True)

    # Rétropropagation couches cachées (de L-1 jusqu'à 1)
    for l in range(L - 1, 0, -1):
        dA = xp.dot(params[f"W{l + 1}"].T, dZ)
        dZ = dA * _derivee(cache[f"Z{l}"], activation)
        grads[f"dW{l}"] = (1 / n) * xp.dot(dZ, cache[f"A{l - 1}"].T)
        grads[f"db{l}"] = (1 / n) * xp.sum(dZ, axis=1, keepdims=True)

    return grads
