import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivee_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)