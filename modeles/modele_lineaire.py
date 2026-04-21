import numpy as np



def matrices(n_x , n_y):
    A = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    return A, b

def fonction_score(X , A, b):
    o = np.dot(A, X) + b
    
    return o
    

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def log_loss(Y, P):
   
    n = Y.shape[1]
   
    loss = - 1 / n * np.sum(Y * np.log(P)) 
    return loss

def gradient(X, Y, P):
   
    n = X.shape[1]
    
   
    
    dA = (1/n) * np.dot(P - Y, X.T)
    
   
    db = (1/n) * np.sum(P - Y, axis=1, keepdims=True)
    
    return dA, db


