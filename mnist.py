import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
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


X_train, y_train, X_test, y_test = load_images_from_folder("IMAGES/GROUPS")

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)    

X_train = X_train / 255.0
X_test = X_test / 255.0 


print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)
