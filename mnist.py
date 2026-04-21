import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import modeles.modele_lineaire as ml


def load_images_from_folder(folder_path):
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for img_path in sorted(glob.glob(f"{folder_path}/*.png")):
       
        img = np.array(Image.open(img_path).convert('L'))
        filename = os.path.basename(img_path)
        
        if '-' in filename:
            label = int(filename.split('-')[1].split('_')[0])
        else:
            continue
            
        
        for y in range(0, 25 * 28, 28): 
            for x in range(0, 40 * 28, 28):
                sub_img = img[y:y+28, x:x+28] # On découpe du 28x28 
                
                if sub_img.shape == (28, 28):
                    if 'TRAIN' in filename.upper() or '30000' in filename:
                        X_train.append(sub_img.flatten())
                        y_train.append(label)
                    elif 'TESTS' in filename.upper() or '40000' in filename:
                        X_test.append(sub_img.flatten())
                        y_test.append(label)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Chargement + normalisation
X_train, y_train, X_test, y_test = load_images_from_folder("IMAGES/GROUPS")

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)    

X_train = X_train / 255.0
X_test = X_test / 255.0 


print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)

# on veut une sortie de label sous la forme de vecteur

def vector_label(y, n_classes=10):
    vectors = np.zeros((y.size, n_classes))
    vectors[np.arange(y.size), y] = 1
    return vectors


# Entraînement des données avec le modèle linéaire

Y_train = vector_label(y_train)

A, b = ml.matrices(784, 10)
learning_rate = 0.1
iterations = 1000


def gradient_descent(X, Y, A, b, learning_rate):
    m = X.shape[1]
    
    Z = ml.fonction_score(X, A, b)
    A_pred = ml.softmax(Z)
    
    dZ = A_pred - Y
    dA = np.dot(dZ, X.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    
    A -= learning_rate * dA
    b -= learning_rate * db
    
    return A, b
