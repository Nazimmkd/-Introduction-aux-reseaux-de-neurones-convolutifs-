import numpy as np
import modeles.modele_lineaire as ml
import modeles.modele_couches_cachées as mcc


def train_model(X, Y, model_type="linear", n_h1=64, n_h2=None,
                couches_cachees=None, activation="relu",
                lr=0.1, iters=100, batch_size=256):
    """
    Entraîne un modèle linéaire ou MLP par mini-batches SGD.

    X : (D, N) — numpy ou CuPy
    Y : (C, N) one-hot

    Architecture MLP — deux syntaxes équivalentes :
      couches_cachees=[128, 64]       → [n_x, 128, 64, n_y]   (nouveau)
      n_h1=64, n_h2=32               → [n_x, 64, 32, n_y]     (compat.)
      n_h1=64                        → [n_x, 64, n_y]          (compat.)

    activation : "sigmoid" | "tanh" | "relu" | "heaviside"
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]

    if model_type == "linear":
        W, b = ml.matrices(n_x, n_y)
        params = [W, b]
        desc = "Linéaire"
    else:
        if couches_cachees is not None:
            couches = [n_x] + list(couches_cachees) + [n_y]
        elif n_h2 is not None:
            couches = [n_x, n_h1, n_h2, n_y]
        else:
            couches = [n_x, n_h1, n_y]
        params = mcc.initialisation_mlp(couches, activation)
        desc = f"MLP {couches[1:-1]} — {activation}"

    print(f"\nEntrainement : {desc}...")
    n = X.shape[1]
    L = len(params) // 2 if model_type != "linear" else None
    history = []

    for i in range(iters):
        indices    = np.random.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        for start in range(0, n, batch_size):
            X_batch = X_shuffled[:, start:start + batch_size]
            Y_batch = Y_shuffled[:, start:start + batch_size]

            if model_type == "linear":
                Z     = ml.fonction_score(X_batch, params[0], params[1])
                P     = ml.softmax(Z)
                grads = ml.gradient(X_batch, Y_batch, P)
                params[0] -= lr * grads[0]
                params[1] -= lr * grads[1]
            else:
                cache = mcc.forward_pass(X_batch, params, activation)
                grads = mcc.backpropagation(X_batch, Y_batch, cache, params, activation)
                for key in params:
                    params[key] -= lr * grads["d" + key]

        # Perte sur tout le train après chaque époque
        if model_type == "linear":
            Z      = ml.fonction_score(X, params[0], params[1])
            A_full = ml.softmax(Z)
        else:
            cache  = mcc.forward_pass(X, params, activation)
            A_full = cache[f"A{L}"]

        history.append(float(ml.log_loss(Y, A_full)))

        if (i + 1) % 10 == 0:
            print(f"  [{desc}] iter {i + 1}/{iters}, perte: {history[-1]:.4f}")

    return params, history


def _augment_batch(X_t):
    """
    Augmentation vectorisée pour CIFAR-10 (sans boucle Python).
    X_t : tenseur float32 (N, C, 32, 32) dans [0, 1]
    """
    import torch
    import torch.nn.functional as F

    N, C, H, W = X_t.shape

    mask = torch.rand(N) > 0.5
    X_t  = X_t.clone()
    X_t[mask] = torch.flip(X_t[mask], dims=[3])

    pad = 4
    X_p = F.pad(X_t, (pad, pad, pad, pad), mode='reflect')
    t   = torch.randint(0, 2 * pad + 1, (N,))
    l   = torch.randint(0, 2 * pad + 1, (N,))
    n_i = torch.arange(N).view(N, 1, 1, 1).expand(N, C, H, W)
    c_i = torch.arange(C).view(1, C, 1, 1).expand(N, C, H, W)
    h_i = (torch.arange(H).view(1, 1, H, 1) + t.view(N, 1, 1, 1)).expand(N, C, H, W)
    w_i = (torch.arange(W).view(1, 1, 1, W) + l.view(N, 1, 1, 1)).expand(N, C, H, W)
    X_t = X_p[n_i, c_i, h_i, w_i]

    factor = 1.0 + (torch.rand(N, 1, 1, 1) - 0.5) * 0.4
    X_t    = torch.clamp(X_t * factor, 0.0, 1.0)
    mean   = X_t.mean(dim=(2, 3), keepdim=True)
    c_fac  = 1.0 + (torch.rand(N, 1, 1, 1) - 0.5) * 0.3
    X_t    = torch.clamp((X_t - mean) * c_fac + mean, 0.0, 1.0)

    return X_t


def _augment_batch_medical(X_t, angle=15.0, zoom=0.12, brightness=0.25, contrast=0.25):
    """
    Augmentation spécialisée pour images médicales en niveaux de gris (N, 1, H, W).

    Transformations valides pour des crops de masses mammaires :
      - Flip horizontal  : une masse peut être vue des deux côtés
      - Rotation ±angle° : les masses n'ont pas d'orientation fixe
      - Zoom ±zoom       : variation d'échelle de numérisation
      - Brightness/Contrast : simule différentes calibrations scanner

    NON appliqués : flip vertical (haut/bas anatomiquement significatif),
                    crop agressif (risque d'effacer la masse).
    """
    import torch
    import torch.nn.functional as F
    import math

    N = X_t.shape[0]
    X_t = X_t.clone()

    # Flip horizontal (p=0.5)
    mask = torch.rand(N) > 0.5
    X_t[mask] = torch.flip(X_t[mask], dims=[3])

    # Rotation + zoom combinés en une seule transformation affine (vectorisé)
    angles_deg = (torch.rand(N) - 0.5) * 2 * angle
    zooms      = 1.0 + (torch.rand(N) - 0.5) * 2 * zoom
    a          = angles_deg * (math.pi / 180)
    cos_a, sin_a = torch.cos(a), torch.sin(a)
    theta = torch.zeros(N, 2, 3)
    theta[:, 0, 0] =  cos_a / zooms
    theta[:, 0, 1] = -sin_a / zooms
    theta[:, 1, 0] =  sin_a / zooms
    theta[:, 1, 1] =  cos_a / zooms
    grid = F.affine_grid(theta, X_t.shape, align_corners=False)
    X_t  = F.grid_sample(X_t, grid, align_corners=False,
                          mode='bilinear', padding_mode='reflection')

    # Brightness multiplicatif
    b   = 1.0 + (torch.rand(N, 1, 1, 1) - 0.5) * 2 * brightness
    X_t = torch.clamp(X_t * b, 0.0, 1.0)

    # Contrast autour de la moyenne locale
    mean  = X_t.mean(dim=(2, 3), keepdim=True)
    c_fac = 1.0 + (torch.rand(N, 1, 1, 1) - 0.5) * 2 * contrast
    X_t   = torch.clamp((X_t - mean) * c_fac + mean, 0.0, 1.0)

    return X_t


def train_model_cnn_binaire(X, y, taille=128, lr=1e-4, iters=40, batch_size=8, patience=8,
                            architecture='cnn_binaire', accumulate_steps=4):
    """
    Entraîne un CNN binaire (bénin/malin) sur des images grayscale.
    X : numpy (N, H, W) | y : numpy labels entiers (N,) avec 0=bénin, 1=malin

    architecture      : 'cnn_binaire' — CNN entraîné de zéro
                        'resnet18'    — ResNet18 pré-entraîné ImageNet (transfer learning)
    accumulate_steps  : accumulation de gradients — batch effectif = batch_size × accumulate_steps.
                        Permet de simuler un grand batch (ex: 8×4=32) sans saturer la VRAM.

    Gère le déséquilibre des classes via des poids inversement proportionnels à la fréquence.
    """
    import torch
    import torch.nn as nn
    import modeles.modele_convolutif as mc

    eff_batch = batch_size * accumulate_steps
    print(f"\nEntrainement : CNN binaire [{architecture}] ({taille}x{taille}) CBIS-DDSM...")
    print(f"  batch réel : {batch_size} | accumulation : {accumulate_steps} | batch effectif : {eff_batch}")

    n_total = X.shape[0]
    n_val   = max(1, n_total // 10)
    perm    = np.random.permutation(n_total)
    X_val, y_val = X[perm[:n_val]],  y[perm[:n_val]]
    X_tr,  y_tr  = X[perm[n_val:]], y[perm[n_val:]]
    print(f"  Train : {len(y_tr)} | Validation : {len(y_val)}")

    # Poids de classe inversement proportionnels à la fréquence
    comptes = np.bincount(y_tr, minlength=2)
    poids   = len(y_tr) / (2 * np.maximum(comptes, 1))
    print(f"  Class weights : benin={poids[0]:.2f}, malin={poids[1]:.2f}")

    if architecture == 'resnet18':
        params = mc.init_cnn_transfert(taille)
    else:
        params = mc.init_cnn_binaire(taille)
    model     = params['model']
    device    = params['device']

    # Exclure γ/β de BatchNorm du weight decay : pénaliser ces params drive γ→0
    bn_ids     = {id(p) for m in model.modules()
                  if isinstance(m, torch.nn.BatchNorm2d) for p in m.parameters()}
    params_wd  = [p for p in model.parameters() if id(p) not in bn_ids]
    params_nwd = [p for p in model.parameters() if id(p) in bn_ids]
    optimizer = torch.optim.Adam([
        {'params': params_wd,  'weight_decay': 1e-4},
        {'params': params_nwd, 'weight_decay': 0.0},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(poids, dtype=torch.float32).to(device))

    n = len(y_tr)
    hist_train, hist_val = [], []
    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0

    for i in range(iters):
        model.train(True)
        indices  = np.random.permutation(n)
        X_shuf   = X_tr[indices]
        y_shuf   = y_tr[indices]
        loss_sum, n_seen = 0.0, 0

        n_batches = (n + batch_size - 1) // batch_size
        optimizer.zero_grad()

        for step_idx, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            X_t = torch.tensor(X_shuf[start:end], dtype=torch.float32).unsqueeze(1)
            X_t = _augment_batch_medical(X_t).to(device)
            y_t = torch.tensor(y_shuf[start:end], dtype=torch.long).to(device)

            # Division par accumulate_steps pour que la moyenne des gradients
            # soit équivalente à un vrai batch de taille batch_size×accumulate_steps
            loss = criterion(model(X_t), y_t)
            (loss / accumulate_steps).backward()
            loss_sum += loss.item() * (end - start)
            n_seen   += end - start

            last_batch = (step_idx + 1 == n_batches)
            if (step_idx + 1) % accumulate_steps == 0 or last_batch:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = loss_sum / n_seen
        model.train(False)
        val_loss, val_seen, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(y_val), batch_size):
                end    = min(start + batch_size, len(y_val))
                X_v    = torch.tensor(X_val[start:end], dtype=torch.float32).unsqueeze(1).to(device)
                y_v    = torch.tensor(y_val[start:end], dtype=torch.long).to(device)
                logits = model(X_v)
                val_loss    += float(criterion(logits, y_v)) * (end - start)
                val_correct += (logits.argmax(dim=1) == y_v).sum().item()
                val_seen    += end - start
        val_loss /= val_seen
        val_acc   = val_correct / val_seen * 100

        scheduler.step(val_loss)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"  [CNN bin] iter {i+1:>3}/{iters} — Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | Val acc: {val_acc:.1f}% | LR: {lr_cur:.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  [CNN bin] Early stopping à l'époque {i+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return params, hist_train, hist_val


def _mixup_batch(X_t, alpha=0.4):
    """
    MixUp (Zhang et al., 2018) : mélange convexe de deux images et de leurs étiquettes.
    Paramètre lambda ~ Beta(alpha, alpha) :
      alpha proche de 0 → lambda ≈ 0 ou 1 (peu de mélange)
      alpha = 0.4–1.0  → mélange significatif

    Retourne X_mixé, indices de permutation, lambda (float).
    La perte doit être calculée comme :
      loss = lambda * CE(logits, y_a) + (1-lambda) * CE(logits, y_b)
    """
    import torch
    import numpy as np
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(X_t.shape[0], device=X_t.device)
    return lam * X_t + (1 - lam) * X_t[perm], perm, lam


def train_ablation_cnn(X, y, taille=128, config=None, lr=1e-4, iters=25,
                       batch_size=8, accumulate_steps=4):
    """
    Entraîne une configuration de CNN pour l'étude d'ablation.

    config : dict avec les clés :
      use_bn       : True/False — BatchNorm après chaque conv
      use_dropout  : True/False — Dropout(0.5/0.3) dans le classificateur
      weight_decay : float — coefficient L2 pour Adam (0 = pas de régularisation)
      n_classes    : 2 → CrossEntropyLoss + softmax
                     1 → BCEWithLogitsLoss + sigmoid

    Retourne : params, hist_train, hist_val
    """
    import torch
    import torch.nn as nn
    import modeles.modele_convolutif as mc

    if config is None:
        config = {}
    use_bn       = config.get('use_bn',       True)
    use_dropout  = config.get('use_dropout',  True)
    weight_decay = config.get('weight_decay', 1e-4)
    n_classes    = config.get('n_classes',    2)
    bn_momentum  = config.get('bn_momentum',  0.1)
    use_mixup    = config.get('use_mixup',    False)
    mixup_alpha  = config.get('mixup_alpha',  0.4)
    nom          = config.get('nom',          'CNN ablation')

    print(f"\n{'─' * 55}")
    print(f"ABLATION : {nom}")
    print(f"{'─' * 55}")

    n_total = X.shape[0]
    n_val   = max(1, n_total // 10)
    perm    = np.random.permutation(n_total)
    X_val, y_val = X[perm[:n_val]],  y[perm[:n_val]]
    X_tr,  y_tr  = X[perm[n_val:]], y[perm[n_val:]]
    print(f"  Train : {len(y_tr)} | Val : {len(y_val)}")

    comptes = np.bincount(y_tr, minlength=2)
    poids   = len(y_tr) / (2 * np.maximum(comptes, 1))

    params    = mc.init_cnn_configurable(taille, use_bn, use_dropout, n_classes, bn_momentum)
    model     = params['model']
    device    = params['device']

    # Exclure γ/β de BatchNorm du weight decay : pénaliser ces params drive γ→0,
    # ce qui effondre la normalisation et cause les pertes explosives observées.
    bn_ids     = {id(p) for m in model.modules()
                  if isinstance(m, torch.nn.BatchNorm2d) for p in m.parameters()}
    params_wd  = [p for p in model.parameters() if id(p) not in bn_ids]
    params_nwd = [p for p in model.parameters() if id(p) in bn_ids]
    optimizer = torch.optim.Adam([
        {'params': params_wd,  'weight_decay': weight_decay},
        {'params': params_nwd, 'weight_decay': 0.0},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    if n_classes == 1:
        # pos_weight = n_bénins / n_malins — pénalise davantage les faux négatifs
        pos_weight = torch.tensor([poids[1] / max(poids[0], 1e-8)],
                                  dtype=torch.float32).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(poids, dtype=torch.float32).to(device))

    n              = len(y_tr)
    hist_train, hist_val = [], []
    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    patience_limit   = 6

    for i in range(iters):
        model.train(True)
        idx    = np.random.permutation(n)
        X_shuf = X_tr[idx]
        y_shuf = y_tr[idx]
        loss_sum, n_seen = 0.0, 0
        n_batches = (n + batch_size - 1) // batch_size
        optimizer.zero_grad()

        for step, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            X_t = torch.tensor(X_shuf[start:end], dtype=torch.float32).unsqueeze(1)
            X_t = _augment_batch_medical(X_t).to(device)

            if n_classes == 1:
                y_t = torch.tensor(y_shuf[start:end], dtype=torch.float32).to(device)
                if use_mixup:
                    X_t, perm, lam = _mixup_batch(X_t, mixup_alpha)
                    y_mix = lam * y_t + (1 - lam) * y_t[perm]
                    loss  = criterion(model(X_t).squeeze(1), y_mix)
                else:
                    loss = criterion(model(X_t).squeeze(1), y_t)
            else:
                y_t = torch.tensor(y_shuf[start:end], dtype=torch.long).to(device)
                if use_mixup:
                    X_t, perm, lam = _mixup_batch(X_t, mixup_alpha)
                    logits = model(X_t)
                    # Perte MixUp : combinaison linéaire sur les deux labels durs
                    loss = lam * criterion(logits, y_t) + (1 - lam) * criterion(logits, y_t[perm])
                else:
                    loss = criterion(model(X_t), y_t)

            (loss / accumulate_steps).backward()
            loss_sum += loss.item() * (end - start)
            n_seen   += end - start

            if (step + 1) % accumulate_steps == 0 or (step + 1 == n_batches):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = loss_sum / n_seen

        # ── Validation ──────────────────────────────────────────
        model.train(False)
        val_loss_sum, val_seen, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(y_val), batch_size):
                end = min(start + batch_size, len(y_val))
                X_v = torch.tensor(X_val[start:end], dtype=torch.float32).unsqueeze(1).to(device)
                if n_classes == 1:
                    y_v    = torch.tensor(y_val[start:end], dtype=torch.float32).to(device)
                    logits = model(X_v).squeeze(1)
                    vloss  = criterion(logits, y_v)
                    preds  = (torch.sigmoid(logits) >= 0.5).long()
                    val_correct += (preds == y_v.long()).sum().item()
                else:
                    y_v    = torch.tensor(y_val[start:end], dtype=torch.long).to(device)
                    logits = model(X_v)
                    vloss  = criterion(logits, y_v)
                    preds  = logits.argmax(dim=1)
                    val_correct += (preds == y_v).sum().item()
                val_loss_sum += float(vloss) * (end - start)
                val_seen     += end - start

        val_loss = val_loss_sum / val_seen
        val_acc  = val_correct / val_seen * 100
        scheduler.step(val_loss)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        print(f"  iter {i+1:>2}/{iters} — Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | Val acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping à l'époque {i + 1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return params, hist_train, hist_val


def train_model_cnn(X, y, lr=0.001, iters=40, batch_size=512, patience=6):
    """
    Entraîne le CNN CIFAR-10 via PyTorch avec early stopping.
    X : numpy (N, 32, 32, 3) | y : numpy labels entiers (N,)
    """
    import torch
    import torch.nn as nn
    import modeles.modele_convolutif as mc

    print(f"\nEntrainement : CNN (PyTorch)...")

    n_total = X.shape[0]
    n_val   = n_total // 10
    perm    = np.random.permutation(n_total)
    X_val, y_val = X[perm[:n_val]],  y[perm[:n_val]]
    X_tr,  y_tr  = X[perm[n_val:]], y[perm[n_val:]]
    print(f"  Train : {len(y_tr)} exemples | Validation : {len(y_val)} exemples")

    params    = mc.init_cnn()
    model     = params['model']
    device    = params['device']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    n                = X_tr.shape[0]
    hist_train, hist_val = [], []
    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0

    for i in range(iters):
        model.train(True)
        indices   = np.random.permutation(n)
        X_shuf    = X_tr[indices]
        y_shuf    = y_tr[indices]
        n_batches = (n + batch_size - 1) // batch_size
        loss_sum, n_seen = 0.0, 0

        for batch_idx, start in enumerate(range(0, n, batch_size)):
            end  = min(start + batch_size, n)
            X_t  = torch.tensor(X_shuf[start:end], dtype=torch.float32).permute(0, 3, 1, 2)
            X_t  = _augment_batch(X_t).to(device)
            y_t  = torch.tensor(y_shuf[start:end], dtype=torch.long).to(device)

            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * (end - start)
            n_seen   += end - start

            if batch_idx % 100 == 0:
                print(f"  [CNN] iter {i+1}/{iters}, batch {batch_idx+1}/{n_batches}, loss: {loss.item():.4f}")

        train_loss = loss_sum / n_seen
        model.train(False)
        val_loss, val_seen, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(y_val), batch_size):
                end    = min(start + batch_size, len(y_val))
                X_v    = torch.tensor(X_val[start:end], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                y_v    = torch.tensor(y_val[start:end], dtype=torch.long).to(device)
                logits = model(X_v)
                val_loss    += float(criterion(logits, y_v)) * (end - start)
                val_correct += (logits.argmax(dim=1) == y_v).sum().item()
                val_seen    += end - start
        val_loss /= val_seen
        val_acc   = val_correct / val_seen * 100

        scheduler.step(val_loss)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[CNN] iter {i+1:>3}/{iters} — Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val acc: {val_acc:.1f}% | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[CNN] Early stopping à l'époque {i+1} — meilleure val loss: {best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[CNN] Meilleurs poids restaurés (val loss: {best_val_loss:.4f})")

    return params, hist_train, hist_val
