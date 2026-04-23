import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import modeles.modele_lineaire as ml
import modeles.modele_couches_cachées as mcc

def load_images_from_folder(folder_path):
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for img_path in sorted(glob.glob(f"{folder_path}/*.png")):
       
        img = np.array(Image.open(img_path).convert('L'))
        filename = os.path.basename(img_path)
        
        # Extraire le label : cherche "TRAIN-0" ou "-5" etc
        label = None
        for part in filename.split('-'):
            # Cherche un nombre d'un seul chiffre (0-9)
            num_str = part.split('_')[0]
            if num_str.isdigit() and 0 <= int(num_str) <= 9:
                label = int(num_str)
                break
        
        if label is None:
            continue
            
        
        for y in range(0, 25 * 28, 28): 
            for x in range(0, 40 * 28, 28):
                sub_img = img[y:y+28, x:x+28] # On découpe du 28x28 
                
                if sub_img.shape == (28, 28):
                    if 'TRAIN' in filename.upper():
                        X_train.append(sub_img.flatten())
                        y_train.append(label)
                    elif 'TESTS' in filename.upper():
                        X_test.append(sub_img.flatten())
                        y_test.append(label)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Chargement + normalisation
X_train, y_train, X_test, y_test = load_images_from_folder("IMAGES/GROUPS")

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)    

X_train = X_train / 255.0
X_test = X_test / 255.0 

# Transposer pour avoir (n_features, n_samples)
X_train = X_train.T
X_test = X_test.T

print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)

# on veut une sortie de label sous la forme de vecteur

def vector_label(y, n_classes=10):
    vectors = np.zeros((n_classes, y.size))
    vectors[y, np.arange(y.size)] = 1
    return vectors


# Entraînement des données avec le modèle linéaire

Y_train = vector_label(y_train)

A, b = ml.matrices(784, 10)
learning_rate = 0.1
iterations = 1000


for i in range(iterations):
    Z = ml.fonction_score(X_train, A, b)
    P = ml.softmax(Z)
    erreur = ml.log_loss(Y_train, P)
    dA, db = ml.gradient(X_train, Y_train, P)

    A = A - learning_rate * dA
    b = b - learning_rate * db
    
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{iterations}, Erreur: {erreur:.4f}")
    

# Entrainement des données avec le modèle à couches cachées


