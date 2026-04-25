import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import os
import modeles.modele_lineaire as ml
import modeles.modele_couches_cachées as mcc


def load_mnist_data():
    data_dir = "data_models"
    os.makedirs(data_dir, exist_ok=True)

    X_train_path = os.path.join(data_dir, "X_train.npy")
    X_test_path  = os.path.join(data_dir, "X_test.npy")
    y_train_path = os.path.join(data_dir, "y_train.npy")
    y_test_path  = os.path.join(data_dir, "y_test.npy")

    if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[LOAD] Chargement des donnees MNIST existantes...")
        X_train = np.load(X_train_path)
        X_test  = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test  = np.load(y_test_path)
    else:
        print("[DOWNLOAD] Telechargement des donnees MNIST...")
        import urllib.request
        import gzip

        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }

        def download_and_parse_images(url):
            with urllib.request.urlopen(url) as r:
                with gzip.open(r) as f:
                    f.read(16)
                    return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784) / 255.0

        def download_and_parse_labels(url):
            with urllib.request.urlopen(url) as r:
                with gzip.open(r) as f:
                    f.read(8)
                    return np.frombuffer(f.read(), dtype=np.uint8).astype(int)

        X_train = download_and_parse_images(base_url + files["train_images"])
        y_train = download_and_parse_labels(base_url + files["train_labels"])
        X_test  = download_and_parse_images(base_url + files["test_images"])
        y_test  = download_and_parse_labels(base_url + files["test_labels"])

        np.save(X_train_path, X_train)
        np.save(X_test_path,  X_test)
        np.save(y_train_path, y_train)
        np.save(y_test_path,  y_test)

    return X_train.T, y_train, X_test.T, y_test


def vector_label(y, n_classes=10):
    assert np.all((y >= 0) & (y < n_classes)), f"Label hors bornes : {np.unique(y)}"
    vectors = np.zeros((n_classes, y.size))
    vectors[y, np.arange(y.size)] = 1
    return vectors


def train_model(X, Y, model_type="linear", n_h1=64, n_h2=None, lr=0.1, iters=100, batch_size=256):
    print(f"\nEntrainement : {model_type}...")

    if model_type == "linear":
        params = list(ml.matrices(784, 10))
    elif n_h2 is None:
        params = mcc.initialisation_mlp(784, n_h1, 10)
    else:
        params = mcc.initialisation_mlp(784, n_h1, 10, n_h2=n_h2)

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
            print(f"[{model_type.upper()}] Epoque {i + 1}/{iters}, Erreur: {loss:.4f}")

    return params, history


def analyze_errors(X, y, params, model_type="linear", n_h2=False):
    if model_type == "linear":
        Z = ml.fonction_score(X, params[0], params[1])
        scores = ml.softmax(Z)
    else:
        out = mcc.forward_pass(X, params)
        scores = out["A3"] if n_h2 else out["A2"]

    predictions = np.argmax(scores, axis=0)
    probs = np.max(scores, axis=0)
    errors = predictions != y
    error_indices = np.where(errors)[0]
    accuracy = np.mean(predictions == y)

    label = f"[MODELE {model_type.upper()} {('H=2' if n_h2 else 'H=1') if model_type == 'mlp' else ''}]"
    print(f"\n{label} (Ensemble TEST)")
    print(f"Precision: {accuracy * 100:.2f}%")
    print(f"Nombre d'erreurs: {len(error_indices)} ({len(error_indices) / len(y) * 100:.2f}%)")
    print(f"Predictions correctes: {len(y) - len(error_indices)} ({(1 - len(error_indices) / len(y)) * 100:.2f}%)")

    if len(error_indices) > 0:
        print("\nExemples d'erreurs:")
        for i in error_indices[:5]:
            print(f"  Indice {i}: Vraie classe={y[i]}, Prediction={predictions[i]}, Confiance={probs[i] * 100:.2f}%")

    return predictions, accuracy


def visualiser_exemples(X_train, y_train):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Exemples MNIST', fontsize=16)
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        ax = axes[i // 5, i % 5]
        ax.imshow(X_train[:, idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualiser_erreurs(X_test, y_test, predictions, model_name="MLP"):
    error_indices = np.where(predictions != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"Exemples mal classes par {model_name}", fontsize=13)
    for i, idx in enumerate(error_indices[:10]):
        ax = axes[i // 5][i % 5]
        ax.imshow(X_test[:, idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Vrai:{y_test[idx]} Predit:{predictions[idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


X_train, y_train, X_test, y_test = load_mnist_data()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Labels uniques: {np.unique(y_train)}")
print(f"Distribution: {np.bincount(y_train)}")

visualiser_exemples(X_train, y_train)

Y_train = vector_label(y_train)
Y_test  = vector_label(y_test)

params_lin, hist_lin = train_model(X_train, Y_train, "linear", lr=0.1, iters=100, batch_size=256)
params_h1,  hist_h1  = train_model(X_train, Y_train, "mlp", n_h1=64, lr=0.1, iters=100, batch_size=256)
params_h2,  hist_h2  = train_model(X_train, Y_train, "mlp", n_h1=64, n_h2=32, lr=0.1, iters=100, batch_size=256)

print("\n" + "=" * 60)
print("EVALUATION ET ANALYSE DES ERREURS DE CLASSIFICATION")
print("=" * 60)

pred_lin, acc_lin = analyze_errors(X_test, y_test, params_lin, "linear")
pred_h1,  acc_h1  = analyze_errors(X_test, y_test, params_h1,  "mlp")
pred_h2,  acc_h2  = analyze_errors(X_test, y_test, params_h2,  "mlp", n_h2=True)

print("\n" + "=" * 60)
print(f"{'Modele':<20} {'Precision Test':>15}")
print("-" * 40)
print(f"{'Lineaire':<20} {acc_lin * 100:>14.2f}%")
print(f"{'MLP H=1':<20} {acc_h1  * 100:>14.2f}%")
print(f"{'MLP H=2':<20} {acc_h2  * 100:>14.2f}%")

visualiser_erreurs(X_test, y_test, pred_h1, "MLP H=1")

print("\nCalcul de la PCA en cours...")
pca = PCA(n_components=2)
pca.fit(X_train.T)
X_test_2d = pca.transform(X_test.T)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_lin, label="Lineaire")
plt.plot(hist_h1,  label="MLP H=1")
plt.plot(hist_h2,  label="MLP H=2")
plt.title("Convergence (Log Loss)")
plt.xlabel("Epoques")
plt.ylabel("Erreur (Log Loss)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='tab10', s=5, alpha=0.5)
plt.title("Visualisation PCA des chiffres")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.colorbar(scatter, label="Classes (0-9)")
plt.grid(True)

plt.tight_layout()
plt.show()