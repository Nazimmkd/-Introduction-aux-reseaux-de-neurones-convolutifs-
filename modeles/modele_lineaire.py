import numpy as np



def matrices(n_x , n_y):
    A = np.random.rand(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    return A, b

def fonction_score(X , A, b):
    return np.dot(A, X) + b

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)