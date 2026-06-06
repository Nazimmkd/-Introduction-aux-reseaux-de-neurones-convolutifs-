import numpy as np
import os

def charger_mnist():
    data_dir = "data_models"
    os.makedirs(data_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(data_dir, "X_train.npy"),
        "X_test":  os.path.join(data_dir, "X_test.npy"),
        "y_train": os.path.join(data_dir, "y_train.npy"),
        "y_test":  os.path.join(data_dir, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[LOAD] Chargement des données MNIST existantes...")
        X_train = np.load(paths["X_train"])
        X_test  = np.load(paths["X_test"])
        y_train = np.load(paths["y_train"])
        y_test  = np.load(paths["y_test"])
    else:
        print("[DOWNLOAD] Téléchargement des données MNIST...")
        import urllib.request
        import gzip

        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        fichiers = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }

        def _lire_images(url):
            with urllib.request.urlopen(url) as r:
                with gzip.open(r) as f:
                    f.read(16)
                    return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784) / 255.0

        def _lire_labels(url):
            with urllib.request.urlopen(url) as r:
                with gzip.open(r) as f:
                    f.read(8)
                    return np.frombuffer(f.read(), dtype=np.uint8).astype(int)

        X_train = _lire_images(base_url + fichiers["train_images"])
        y_train = _lire_labels(base_url + fichiers["train_labels"])
        X_test  = _lire_images(base_url + fichiers["test_images"])
        y_test  = _lire_labels(base_url + fichiers["test_labels"])

        np.save(paths["X_train"], X_train)
        np.save(paths["X_test"],  X_test)
        np.save(paths["y_train"], y_train)
        np.save(paths["y_test"],  y_test)

    # Retourne en format colonne (D, N) pour compatibilité avec les modèles
    return X_train.T, y_train, X_test.T, y_test


def charger_cifar10():
    import urllib.request
    import tarfile

    data_dir = "data_models"
    os.makedirs(data_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(data_dir, "X_train_cifar.npy"),
        "X_test":  os.path.join(data_dir, "X_test_cifar.npy"),
        "y_train": os.path.join(data_dir, "y_train_cifar.npy"),
        "y_test":  os.path.join(data_dir, "y_test_cifar.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[LOAD] Chargement des données CIFAR-10 existantes...")
        X_train = np.load(paths["X_train"])
        X_test  = np.load(paths["X_test"])
        y_train = np.load(paths["y_train"])
        y_test  = np.load(paths["y_test"])
    else:
        print("[DOWNLOAD] Téléchargement des données CIFAR-10...")
        url      = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)

        # pickle est nécessaire ici — format officiel des fichiers CIFAR-10
        import pickle  # noqa: S403

        def _charger_batch(fichier):
            with open(fichier, 'rb') as fo:
                d = pickle.load(fo, encoding='bytes')  # noqa: S301
            return d[b'data'], d[b'labels']

        batches_dir = os.path.join(data_dir, "cifar-10-batches-py")
        X_train, y_train = [], []
        for i in range(1, 6):
            data, labels = _charger_batch(os.path.join(batches_dir, f"data_batch_{i}"))
            X_train.append(data)
            y_train.extend(labels)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)

        X_test, y_test_list = _charger_batch(os.path.join(batches_dir, "test_batch"))
        y_test = np.array(y_test_list)

        # Reshape vers (N, 32, 32, 3) normalisé
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        X_test  = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

        np.save(paths["X_train"], X_train)
        np.save(paths["X_test"],  X_test)
        np.save(paths["y_train"], y_train)
        np.save(paths["y_test"],  y_test)

    return X_train, y_train, X_test, y_test
