import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivee_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def initialisation_mlp(n_x, n_h, n_y, n_h2=None):
    if n_h2 is None:
        # H = 1
        W1 = np.random.randn(n_h, n_x) * np.sqrt(1 / n_x)  # <-- Xavier
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * np.sqrt(1 / n_h)  # <-- Xavier
        b2 = np.zeros((n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    else:
        # H = 2
        W1 = np.random.randn(n_h, n_x) * np.sqrt(1 / n_x)
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_h2, n_h) * np.sqrt(1 / n_h)
        b2 = np.zeros((n_h2, 1))
        W3 = np.random.randn(n_y, n_h2) * np.sqrt(1 / n_h2)
        b3 = np.zeros((n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
def forward_pass(X, params):
    """Forward pass pour MLP avec H=1 ou H=2"""
    # Couche 1 (cachée)
    Z1 = np.dot(params["W1"], X) + params["b1"]
    A1 = sigmoid(Z1)
    
    if "W3" not in params:
        # H = 1
        Z2 = np.dot(params["W2"], A1) + params["b2"]
        A2 = softmax(Z2)
        return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    else:
        # H = 2
        Z2 = np.dot(params["W2"], A1) + params["b2"]
        A2 = sigmoid(Z2)
        Z3 = np.dot(params["W3"], A2) + params["b3"]
        A3 = softmax(Z3)
        return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

def backpropagation(X, Y, cache, params):
    """Backpropagation pour MLP avec H=1 ou H=2"""
    n = X.shape[1]
    
    if "Z3" not in cache:
        # H = 1
        dZ2 = cache["A2"] - Y
        dW2 = (1/n) * np.dot(dZ2, cache["A1"].T)
        db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = np.dot(params["W2"].T, dZ2) * derivee_sigmoid(cache["Z1"])
        dW1 = (1/n) * np.dot(dZ1, X.T)
        db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    else:
        # H = 2
        dZ3 = cache["A3"] - Y
        dW3 = (1/n) * np.dot(dZ3, cache["A2"].T)
        db3 = (1/n) * np.sum(dZ3, axis=1, keepdims=True)
        
        dZ2 = np.dot(params["W3"].T, dZ3) * derivee_sigmoid(cache["Z2"])
        dW2 = (1/n) * np.dot(dZ2, cache["A1"].T)
        db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = np.dot(params["W2"].T, dZ2) * derivee_sigmoid(cache["Z1"])
        dW1 = (1/n) * np.dot(dZ1, X.T)
        db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

