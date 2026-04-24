import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivee_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def initialisation_mlp(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return W1, b1, W2, b2

def forward_pass(X, params):
    # Couche 1 (cachée)
    Z1 = np.dot(params["W1"], X) + params["b1"]
    A1 = sigmoid(Z1)
    
    # Couche 2 (sortie)
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = softmax(Z2)
    
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def backpropagation(X, Y, cache, params):
    n = X.shape[1]
    
    
    dZ2 = cache["A2"] - Y
    dW2 = (1/n) * np.dot(dZ2, cache["A1"].T)
    db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)
    
    
    dZ1 = np.dot(params["W2"].T, dZ2) * d_sigmoid(cache["Z1"])
    dW1 = (1/n) * np.dot(dZ1, X.T)
    db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

