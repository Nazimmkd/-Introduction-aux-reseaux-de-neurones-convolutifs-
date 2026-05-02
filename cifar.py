import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pickle
import urllib.request
import tarfile
import modeles.modele_lineaire as ml
import modeles.modele_couches_cachées as mcc
import modeles.modele_convolutif as mc

def load_cifar10_data():
    data_dir = "data_models"
    os.makedirs(data_dir, exist_ok=True)

    X_train_path = os.path.join(data_dir, "X_train_cifar.npy")
    X_test_path  = os.path.join(data_dir, "X_test_cifar.npy")
    y_train_path = os.path.join(data_dir, "y_train_cifar.npy")
    y_test_path  = os.path.join(data_dir, "y_test_cifar.npy")

    if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[LOAD] Chargement des donnees CIFAR-10 existantes...")
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
    else:
        print("[DOWNLOAD] Telechargement des donnees CIFAR-10...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
        urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        def load_batch(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict[b'data'], dict[b'labels']
        
        data_dir_cifar = os.path.join(data_dir, "cifar-10-batches-py")
        X_train = []
        y_train = []
        for i in range(1, 6):
            data, labels = load_batch(os.path.join(data_dir_cifar, f"data_batch_{i}"))
            X_train.append(data)
            y_train.extend(labels)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)
        
        X_test, y_test_list = load_batch(os.path.join(data_dir_cifar, "test_batch"))
        y_test = np.array(y_test_list)
        
        # Reshape to (N, 32, 32, 3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        
        np.save(X_train_path, X_train)
        np.save(X_test_path,  X_test)
        np.save(y_train_path, y_train)
        np.save(y_test_path,  y_test)

    return X_train, y_train, X_test, y_test

def rgb_to_grayscale(X):
    """Convert RGB images to grayscale."""
    return np.dot(X, [0.299, 0.587, 0.114])

def vector_label(y, n_classes=10):
    """Convertit les labels entiers en vecteurs one-hot."""
    assert np.all((y >= 0) & (y < n_classes)), f"Label hors bornes : {np.unique(y)}"
    vectors = np.zeros((n_classes, y.size))
    vectors[y, np.arange(y.size)] = 1
    return vectors

def train_model(X, Y, model_type="linear", n_h1=64, n_h2=None, lr=0.1, iters=100, batch_size=256):
    print(f"\nEntrainement : {model_type}...")

    if model_type == "linear":
        W, b = ml.matrices(X.shape[0], 10)
        params = [W, b]
    elif n_h2 is None:
        params = mcc.initialisation_mlp(X.shape[0], n_h1, 10)
    else:
        params = mcc.initialisation_mlp(X.shape[0], n_h1, 10, n_h2=n_h2)

    n = X.shape[1]
    history = []

    for i in range(iters):
        indices = np.random.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        for start in range(0, n, batch_size):
            X_batch = X_shuffled[:, start:start + batch_size]
            Y_batch = Y_shuffled[:, start:start + batch_size]

            if model_type == "linear":
                Z = ml.fonction_score(X_batch, params[0], params[1])
                A_out = ml.softmax(Z)
                grads = ml.gradient(X_batch, Y_batch, A_out)
                params[0] -= lr * grads[0]
                params[1] -= lr * grads[1]
            else:
                outputs = mcc.forward_pass(X_batch, params)
                A_out = outputs["A3"] if n_h2 is not None else outputs["A2"]
                grads = mcc.backpropagation(X_batch, Y_batch, outputs, params)
                for key in params:
                    params[key] -= lr * grads["d" + key]

        if model_type == "linear":
            Z = ml.fonction_score(X, params[0], params[1])
            A_full = ml.softmax(Z)
        else:
            outputs = mcc.forward_pass(X, params)
            A_full = outputs["A3"] if n_h2 is not None else outputs["A2"]

        loss = ml.log_loss(Y, A_full)
        history.append(loss)

        if (i + 1) % 10 == 0:
            print(f"[{model_type.upper()}] itération {i + 1}/{iters}, Erreur: {loss:.4f}")

    return params, history

def train_model_cnn(X, Y, lr=0.01, iters=10, batch_size=64):
    print(f"\nEntrainement : CNN...")
    
    params = mc.init_cnn()
    n = X.shape[0]
    history = []
    
    for i in range(iters):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        Y_shuffled = Y[:, indices]
        
        n_batches = (n + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, n, batch_size)):
            X_batch = X_shuffled[start:start + batch_size]
            Y_batch = Y_shuffled[:, start:start + batch_size]
            
            if batch_idx % 4 == 0:
                print(f"  [CNN] itération {i + 1}/{iters}, batch {batch_idx + 1}/{n_batches}")

            A_out, cache = mc.cnn_forward(X_batch, params)
            grads = mc.cnn_backprop(X_batch, Y_batch, cache, params)
            
            # Update params
            for key in params:
                if key in grads:
                    params[key] -= lr * grads["d" + key]
        
        # Compute loss on full batch (approximate)
        A_full, _ = mc.cnn_forward(X[:batch_size], params)
        loss = ml.log_loss(Y[:, :batch_size], A_full)
        history.append(loss)
        
        if (i + 1) % 1 == 0:
            print(f"[CNN] itération {i + 1}/{iters}, Erreur: {loss:.4f}")
    
    return params, history

def analyze_errors_cifar(X, y, params, model_type="mlp", n_h2=False, ensemble="TEST"):
    if model_type == "linear":
        # X is flat (D, N)
        Z = ml.fonction_score(X, params[0], params[1])
        scores = ml.softmax(Z)
    elif model_type == "mlp":
        # X is already flat (D, N)
        out = mcc.forward_pass(X, params)
        scores = out["A3"] if n_h2 else out["A2"]
    elif model_type == "cnn":
        scores, _ = mc.cnn_forward(X, params)
    else:
        raise ValueError(f"Model type inconnu : {model_type}")

    predictions = np.argmax(scores, axis=0)
    probs = np.max(scores, axis=0)
    errors = predictions != y
    error_indices = np.where(errors)[0]
    accuracy = np.mean(predictions == y)

    label = f"[MODELE {model_type.upper()} {('H=2' if n_h2 else 'H=1') if model_type == 'mlp' else ''}]"
    print(f"\n{label} (Ensemble {ensemble})")
    print(f"Precision: {accuracy * 100:.2f}%")
    print(f"Nombre d'erreurs: {len(error_indices)} ({len(error_indices) / len(y) * 100:.2f}%)")

    return predictions, accuracy

def visualiser_exemples_cifar(X_train, y_train):
    class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Exemples CIFAR-10', fontsize=16)
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        ax = axes[i // 5, i % 5]
        ax.imshow(X_train[idx])
        ax.set_title(f"{class_names[i]}")
        ax.axis('off')
    plt.tight_layout()
    print("Fermez la fenêtre d'exemple pour continuer l'entraînement...")
    plt.show()
    print("Fenêtre fermée, reprise du script.")

def visualiser_erreurs_cifar(X_test, y_test, predictions):
    class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    error_indices = np.where(predictions != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Exemples mal classés par le CNN CIFAR-10", fontsize=13)
    for i, idx in enumerate(error_indices[:10]):
        ax = axes[i // 5][i % 5]
        ax.imshow(X_test[idx])
        ax.set_title(f"Vrai: {class_names[y_test[idx]]}\nPredit: {class_names[predictions[idx]]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("cifar_erreurs.png", dpi=300, bbox_inches='tight')
    plt.show()

# Load data
X_train, y_train, X_test, y_test = load_cifar10_data()

# Réduire les jeux de données pour accélérer le débogage du CNN
n_train = 500   # nombre d'exemples d'entraînement à utiliser
n_test = 500    # nombre d'exemples de test à utiliser
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]

# After loading data
visualiser_exemples_cifar(X_train, y_train)

# Convertir en niveaux de gris pour MLP
X_train_gray = rgb_to_grayscale(X_train)
X_test_gray = rgb_to_grayscale(X_test)

# Aplatissement pour MLP
X_train_flat = X_train_gray.reshape(X_train_gray.shape[0], -1).T
X_test_flat = X_test_gray.reshape(X_test_gray.shape[0], -1).T

# Pour MLP en couleur
X_train_color_flat = X_train.reshape(X_train.shape[0], -1).T
X_test_color_flat = X_test.reshape(X_test.shape[0], -1).T

Y_train = vector_label(y_train)
Y_test = vector_label(y_test)

print("Test du MLP sur CIFAR-10 en niveaux de gris")
params_lin_gray, hist_lin_gray = train_model(X_train_flat, Y_train, "linear", lr=0.1, iters=10, batch_size=256)
params_h1_gray, hist_h1_gray = train_model(X_train_flat, Y_train, "mlp", n_h1=128, lr=0.1, iters=10, batch_size=256)

print("\nTest du MLP sur CIFAR-10 en couleur")
params_lin_color, hist_lin_color = train_model(X_train_color_flat, Y_train, "linear", lr=0.1, iters=10, batch_size=256)
params_h1_color, hist_h1_color = train_model(X_train_color_flat, Y_train, "mlp", n_h1=128, lr=0.1, iters=10, batch_size=256)

# Évaluation MLP
pred_lin_gray, acc_lin_gray = analyze_errors_cifar(X_test_flat, y_test, params_lin_gray, "linear", ensemble="TEST")
pred_h1_gray, acc_h1_gray = analyze_errors_cifar(X_test_flat, y_test, params_h1_gray, "mlp", ensemble="TEST")
pred_lin_color, acc_lin_color = analyze_errors_cifar(X_test_color_flat, y_test, params_lin_color, "linear", ensemble="TEST")
pred_h1_color, acc_h1_color = analyze_errors_cifar(X_test_color_flat, y_test, params_h1_color, "mlp", ensemble="TEST")

print("\nEntrainement du CNN...")
params_cnn, hist_cnn = train_model_cnn(X_train, Y_train, lr=0.01, iters=5, batch_size=64)

pred_cnn, acc_cnn = analyze_errors_cifar(X_test, y_test, params_cnn, "cnn", ensemble="TEST")
print(f"\nPrécision CNN: {acc_cnn * 100:.2f}%")

# Graphiques pour le rapport
print("\nGénération des graphiques pour le rapport...")

# Convergence CIFAR
plt.figure(figsize=(10, 6))
plt.plot(hist_lin_gray, label="Linéaire (gris)")
plt.plot(hist_h1_gray, label="MLP H=1 (gris)")
plt.plot(hist_lin_color, label="Linéaire (couleur)")
plt.plot(hist_h1_color, label="MLP H=1 (couleur)")
plt.plot(hist_cnn, label="CNN")
plt.title("Convergence CIFAR-10 (Log Loss)")
plt.xlabel("Itérations")
plt.ylabel("Erreur (Log Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar_convergence.png", dpi=300, bbox_inches='tight')
plt.show()

# ACP CIFAR-10
print("\nGénération de l'ACP CIFAR-10...")
acp_cifar = PCA(n_components=2)
X_train_flat_acp = X_train_gray.reshape(X_train_gray.shape[0], -1)
acp_cifar.fit(X_train_flat_acp)
X_test_2d = acp_cifar.transform(X_test_gray.reshape(X_test_gray.shape[0], -1))

plt.figure(figsize=(10, 8))
class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    mask = y_test == i
    plt.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], c=[colors[i]], label=class_names[i], alpha=0.6, s=10)
plt.title("ACP CIFAR-10 (niveaux de gris)")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar_acp.png", dpi=300, bbox_inches='tight')
plt.show()