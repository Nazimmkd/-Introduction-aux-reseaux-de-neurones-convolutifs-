import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Fonctions éducatives (implémentation NumPy manuelle)
# ──────────────────────────────────────────────────────────────

def convolution2d(X, K, stride=1, padding=0):
    """
    Convolution 2D pour une image en niveaux de gris.
    X: (H, W)
    K: (KH, KW)
    Retourne: (H', W')
    """
    H, W = X.shape
    KH, KW = K.shape
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X
    H_p, W_p = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.sum(X_padded[i * stride:i * stride + KH, j * stride:j * stride + KW] * K)
    return out


def convolution_rgb(X, K, stride=1, padding=0):
    """
    Convolution pour image RGB.
    X: (H, W, 3)
    K: (KH, KW, 3)
    Retourne: (H', W')
    """
    H, W, C = X.shape
    KH, KW, _ = K.shape
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X
    H_p, W_p, _ = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.sum(X_padded[i * stride:i * stride + KH, j * stride:j * stride + KW, :] * K)
    return out


def max_pooling(X, pool_size=2, stride=2):
    """
    Max pooling 2D.
    X: (H, W)
    Retourne: (H', W')
    """
    H, W = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.max(X[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size])
    return out


def flatten(X):
    """
    Aplatit le volume en vecteur.
    X: (H, W, C)
    Retourne: (H*W*C,)
    """
    return X.flatten()


def conv_layer(X, filters, biases, stride=1, padding=0):
    """
    Couche de convolution avec multiple filtres.
    X: (H, W, C_in)
    filters: (num_filters, KH, KW, C_in)
    biases: (num_filters,)
    Retourne: (H', W', num_filters)
    """
    num_filters, KH, KW, C_in = filters.shape
    H, W, C = X.shape
    assert C == C_in
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X
    H_p, W_p, _ = X_padded.shape
    out_H = (H_p - KH) // stride + 1
    out_W = (W_p - KW) // stride + 1
    out = np.zeros((out_H, out_W, num_filters))
    for f in range(num_filters):
        for i in range(out_H):
            for j in range(out_W):
                out[i, j, f] = np.sum(
                    X_padded[i * stride:i * stride + KH, j * stride:j * stride + KW, :] * filters[f]) + biases[f]
    return out


def pool_layer(X, pool_size=2, stride=2):
    """
    Couche de max pooling.
    X: (H, W, C)
    Retourne: (H', W', C)
    """
    H, W, C = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W, C))
    for c in range(C):
        out[:, :, c] = max_pooling(X[:, :, c], pool_size, stride)
    return out


def init_conv_filters(num_filters, KH, KW, C_in):
    return np.random.randn(num_filters, KH, KW, C_in) * 0.01


def init_conv_biases(num_filters):
    return np.zeros(num_filters)


def dense_forward(X, W, b):
    return np.dot(W, X) + b


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def max_pool_backprop(input, grad_output, pool_size=2, stride=2):
    """
    Retropropagation pour max pool.
    input: (H, W)
    grad_output: (H', W')
    """
    H, W = input.shape
    H_out, W_out = grad_output.shape
    grad_input = np.zeros_like(input)
    for i in range(H_out):
        for j in range(W_out):
            window = input[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            max_val = np.max(window)
            mask = (window == max_val)
            grad_input[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size] += mask * grad_output[i, j]
    return grad_input


def conv_backprop(input, output, grad_output, filters, biases, padding=0):
    """
    Retropropagation pour la couche de conv.
    input: (H_in, W_in, C_in)
    output: (H_out, W_out, num_filters)
    grad_output: (H_out, W_out, num_filters)
    filters: (num_filters, KH, KW, C_in)
    biases: (num_filters,)
    """
    num_filters, KH, KW, C_in = filters.shape
    H_in, W_in, C = input.shape
    H_out, W_out, _ = grad_output.shape

    input_padded = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    db = np.sum(grad_output, axis=(0, 1))

    df = np.zeros_like(filters)
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                df[f] += grad_output[i, j, f] * input_padded[i:i + KH, j:j + KW, :]

    dinput_padded = np.zeros_like(input_padded)
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                dinput_padded[i:i + KH, j:j + KW, :] += grad_output[i, j, f] * filters[f]

    if padding > 0:
        dinput = dinput_padded[padding:-padding, padding:-padding, :]
    else:
        dinput = dinput_padded
    return dinput, df, db


def visualiser_filtres_K(image_gray, save_path="filtres_convolution.png"):
    """
    Applique et visualise les 6 filtres du cours (section 2.3.2) sur une image en niveaux de gris.
    image_gray: (H, W) tableau numpy, valeurs dans [0, 1]
    """
    import matplotlib.pyplot as plt

    K1 = (1 / 9) * np.ones((3, 3))
    K2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    K3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float)
    K4 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    K5 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    K6 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float)

    filtres = [
        (K1, "K1 — Flou (moyenneur)"),
        (K2, "K2 — Netteté"),
        (K3, "K3 — Bords horizontaux"),
        (K4, "K4 — Bords verticaux"),
        (K5, "K5 — Sobel horizontal"),
        (K6, "K6 — Diagonal"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Effet des filtres de convolution — Section 2.3.2", fontsize=13)

    axes[0, 0].imshow(image_gray, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')

    for idx, (K, titre) in enumerate(filtres):
        filtered = convolution2d(image_gray, K, padding=1)
        vmin, vmax = filtered.min(), filtered.max()
        filtered_norm = (filtered - vmin) / (vmax - vmin + 1e-8)
        row, col = (idx + 1) // 4, (idx + 1) % 4
        axes[row, col].imshow(filtered_norm, cmap='gray')
        axes[row, col].set_title(titre, fontsize=9)
        axes[row, col].axis('off')

    axes[1, 3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────
# CNN GPU via PyTorch (Architecture exacte du schéma du cours)
# ──────────────────────────────────────────────────────────────

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Bloc 1 ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # --- Bloc 2 ---
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # --- Bloc 3 (La 4ème convolution du schéma) ---
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- Classificateur final (4096 vers 10) ---
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        # 1. Entrée -> Conv1 -> Conv2 -> MaxPool1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # 2. Conv3 -> MaxPool2
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # 3. Conv4
        x = F.relu(self.conv4(x))

        # 4. Aplatissement, Dropout, Classificateur Dense
        x = x.reshape(x.size(0), -1)  # vecteur de 4096
        x = self.dropout(x)           # régularisation contre l'overfitting
        return self.fc(x)             # logits bruts (sans softmax)


def init_cnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [CNN] Device : {device}")
    model = CNNModel().to(device)
    return {'model': model, 'device': device}


# ──────────────────────────────────────────────────────────────
# CNN binaire — images en niveaux de gris, sortie 2 classes
# ──────────────────────────────────────────────────────────────

class CNNBinaire(nn.Module):
    """
    CNN pour classification binaire sur images grayscale de taille taille×taille.
    4 blocs conv+BN+ReLU+Pool, puis deux couches denses.
    BatchNorm améliore la stabilité avec les petits datasets médicaux.
    """
    def __init__(self, n_classes=2, taille=128):
        super().__init__()
        self.bloc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # taille → taille/2
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # taille/2 → taille/4
        )
        self.bloc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # taille/4 → taille/8
        )
        self.bloc4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # taille/8 → taille/16
        )
        dim_fc = 128 * (taille // 16) * (taille // 16)
        self.classificateur = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim_fc, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.bloc4(self.bloc3(self.bloc2(self.bloc1(x))))
        return self.classificateur(x.reshape(x.size(0), -1))


def init_cnn_binaire(taille=128, n_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [CNN binaire] Device : {device} | entree {taille}x{taille} gris -> {n_classes} classes")
    model = CNNBinaire(n_classes=n_classes, taille=taille).to(device)
    return {'model': model, 'device': device, 'taille': taille}


# ──────────────────────────────────────────────────────────────
# Transfer learning — ResNet18 pré-entraîné adapté au grayscale
# ──────────────────────────────────────────────────────────────

class CNNTransfert(nn.Module):
    """
    ResNet18 pré-entraîné sur ImageNet, adapté pour :
    - 1 canal d'entrée (grayscale) au lieu de 3 (RGB)
    - Classification binaire (2 classes)
    - Fine-tuning des 2 derniers blocs uniquement (économise la mémoire)

    Le transfer learning est particulièrement efficace sur les petits datasets
    médicaux : ResNet18 a appris des détecteurs de textures universels.
    """
    def __init__(self, n_classes=2):
        super().__init__()
        try:
            import torchvision.models as tv
            backbone = tv.resnet18(weights='DEFAULT')
        except Exception:
            import torchvision.models as tv
            backbone = tv.resnet18(pretrained=True)

        # Adapter le premier conv pour 1 canal gris
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Geler tous les paramètres sauf layer3, layer4 et fc
        for name, param in backbone.named_parameters():
            if not any(x in name for x in ('layer3', 'layer4', 'fc', 'conv1', 'bn1')):
                param.requires_grad = False

        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def init_cnn_transfert(taille=128, n_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [ResNet18 transfert] Device : {device} | entree {taille}x{taille} gris -> {n_classes} classes")
    model = CNNTransfert(n_classes=n_classes).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametres entrainables : {n_param:,}")
    return {'model': model, 'device': device, 'taille': taille}


def cnn_binaire_forward(X_batch, params):
    """
    Passe avant pour le CNN binaire.
    X_batch : numpy (N, H, W) — images en niveaux de gris
    Retourne : (n_classes, N) numpy
    """
    model  = params['model']
    device = params['device']
    if hasattr(X_batch, 'get'):
        X_batch = X_batch.get()

    model.train(False)
    batch_size = 256
    probs_list = []

    with torch.no_grad():
        for i in range(0, X_batch.shape[0], batch_size):
            X_t = torch.tensor(
                X_batch[i:i + batch_size], dtype=torch.float32
            ).unsqueeze(1).to(device)  # (N,H,W) → (N,1,H,W)
            p = F.softmax(model(X_t), dim=1)
            probs_list.append(p.cpu().numpy())

    return np.concatenate(probs_list, axis=0).T, {}  # (n_classes, N)


def cnn_forward(X_batch, params):
    """
    Passe avant (inference) sécurisée par lots pour éviter de saturer la VRAM.
    X_batch : numpy ou cupy (N, H, W, C)
    Retourne : (A_out numpy (10, N), cache vide)
    """
    model = params['model']
    device = params['device']

    if hasattr(X_batch, 'get'):  # tableau CuPy -> numpy
        X_batch = X_batch.get()

    model.train(False)
    batch_size = 512  # découpe le calcul pour protéger la carte graphique
    probs_list = []

    with torch.no_grad():
        for i in range(0, X_batch.shape[0], batch_size):
            X_t = torch.tensor(X_batch[i:i + batch_size], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            p = F.softmax(model(X_t), dim=1)
            probs_list.append(p.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    return probs.T, {}  # (10, N)


def cnn_backprop(X_batch, Y_batch, cache, params):
    # La rétropropagation est gérée par PyTorch dans train_model_cnn.
    return {}


# ──────────────────────────────────────────────────────────────
# CNN configurable — ablation study BN / Dropout / BCE vs CE
# ──────────────────────────────────────────────────────────────

def _bloc_conv_abl(c_in, c_out, use_bn, double=False, bn_momentum=0.1):
    """
    Construit un bloc Conv(+BN)+ReLU(+Conv(+BN)+ReLU)+MaxPool.
    bn_momentum : momentum des running stats de BatchNorm.
      0.1 (défaut PyTorch) → mise à jour rapide, sensible au bruit de mini-batch.
      0.01 → mise à jour lente, statistiques stables même avec batch_size=8.
    """
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(c_out, momentum=bn_momentum))
    layers.append(nn.ReLU())
    if double:
        layers.append(nn.Conv2d(c_out, c_out, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(c_out, momentum=bn_momentum))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class CNNBinaireConfigurable(nn.Module):
    """
    CNN configurable pour étude d'ablation.

    use_bn      : BatchNorm2d après chaque convolution
    use_dropout : Dropout(0.5) + Dropout(0.3) dans le classificateur
    n_classes   : 2 → CrossEntropyLoss + softmax (2 neurones sortie)
                  1 → BCEWithLogitsLoss + sigmoid (1 neurone sortie)

    Initialisation de He (Kaiming) appliquée explicitement à toutes les couches.
    std = sqrt(2 / fan_out) pour les activations ReLU.
    """
    def __init__(self, taille=128, use_bn=True, use_dropout=True, n_classes=2, bn_momentum=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.blocs = nn.Sequential(
            _bloc_conv_abl(1,   32,  use_bn, double=True, bn_momentum=bn_momentum),
            _bloc_conv_abl(32,  64,  use_bn, bn_momentum=bn_momentum),
            _bloc_conv_abl(64,  128, use_bn, bn_momentum=bn_momentum),
            _bloc_conv_abl(128, 128, use_bn, bn_momentum=bn_momentum),
        )
        dim_fc = 128 * (taille // 16) ** 2
        clf = []
        if use_dropout:
            clf.append(nn.Dropout(0.5))
        clf += [nn.Linear(dim_fc, 256), nn.ReLU()]
        if use_dropout:
            clf.append(nn.Dropout(0.3))
        clf.append(nn.Linear(256, n_classes))
        self.classificateur = nn.Sequential(*clf)
        self._init_kaiming()

    def _init_kaiming(self):
        """
        Initialisation de He (2015) : std = sqrt(2 / fan_out).
        Compense la perte de variance causée par ReLU (qui met la moitié
        des activations à zéro), contrairement à Xavier qui suppose une
        activation linéaire.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)   # γ = 1 (pas de mise à l'échelle initiale)
                nn.init.zeros_(m.bias)    # β = 0 (pas de décalage initial)

    def forward(self, x):
        x = self.blocs(x)
        return self.classificateur(x.reshape(x.size(0), -1))


def init_cnn_configurable(taille=128, use_bn=True, use_dropout=True, n_classes=2, bn_momentum=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bn_desc = f"BN(m={bn_momentum})" if use_bn else "sans BN"
    desc = (f"{bn_desc} | "
            f"Drop={'oui' if use_dropout else 'non'} | "
            f"{'BCE (1 sortie+sigmoid)' if n_classes == 1 else 'CE (2 sorties+softmax)'}")
    print(f"  [CNN configurable] {desc} | {taille}×{taille} gris")
    model = CNNBinaireConfigurable(taille=taille, use_bn=use_bn, use_dropout=use_dropout,
                                    n_classes=n_classes, bn_momentum=bn_momentum).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {n_param:,}")
    return {'model': model, 'device': device, 'taille': taille}


def cnn_configurable_forward(X_batch, params):
    """
    Passe avant pour CNNBinaireConfigurable.
    Gère BCE (n_classes=1, sigmoid) et CE (n_classes=2, softmax).
    Retourne toujours (2, N) pour compatibilité avec evaluer_modele.
    """
    model     = params['model']
    device    = params['device']
    n_classes = model.n_classes
    if hasattr(X_batch, 'get'):
        X_batch = X_batch.get()

    model.train(False)
    probs_list = []
    with torch.no_grad():
        for i in range(0, X_batch.shape[0], 256):
            X_t    = torch.tensor(X_batch[i:i + 256], dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(X_t)
            if n_classes == 1:
                # BCE : sigmoid sur 1 logit → reconstruit [1-p, p] pour compatibilité
                p     = torch.sigmoid(logits.squeeze(1))
                probs = torch.stack([1 - p, p], dim=1)
            else:
                probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    return np.concatenate(probs_list, axis=0).T, {}  # (2, N)