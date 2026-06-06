import numpy as np
import os
from sklearn.decomposition import PCA

import utils.sorties as sorties
from utils.donnees import charger_cifar10
from utils.gpu import to_gpu, to_cpu
from utils.preprocessing import vector_label, rgb_to_grayscale
from utils.entrainement import train_model, train_model_cnn
from utils.evaluation import evaluer_modele
from utils.visualisation import (
    visualiser_exemples,
    visualiser_erreurs,
    tracer_convergence,
    tracer_acp,
    CIFAR_CLASSES,
)
import modeles.modele_convolutif as mc

S = lambda nom: sorties.chemin("cifar",   nom)
M = lambda nom: sorties.chemin("modeles", nom)

CNN_WEIGHTS_PATH = M("cnn_cifar10.pth")

log = sorties.demarrer_log("cifar")
try:
    # ── Données ───────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = charger_cifar10()
    visualiser_exemples(X_train, y_train, titre='Exemples CIFAR-10', class_names=CIFAR_CLASSES)

    X_train_gray = rgb_to_grayscale(X_train)
    X_test_gray  = rgb_to_grayscale(X_test)

    print("\nDémonstration des filtres de convolution K1–K6 (Section 2.3.2)...")
    mc.visualiser_filtres_K(X_test_gray[0], save_path=S("filtres_convolution.png"))

    # Aplatissement + transfert GPU pour MLP/Linéaire
    X_train_flat       = to_gpu(X_train_gray.reshape(X_train_gray.shape[0], -1).T)
    X_test_flat        = to_gpu(X_test_gray.reshape(X_test_gray.shape[0], -1).T)
    X_train_color_flat = to_gpu(X_train.reshape(X_train.shape[0], -1).T)
    X_test_color_flat  = to_gpu(X_test.reshape(X_test.shape[0], -1).T)
    Y_train            = to_gpu(vector_label(y_train))

    # ── Entraînement MLP / Linéaire ───────────────────────────────
    print("\nTest du MLP sur CIFAR-10 en niveaux de gris")
    params_lin_gray, hist_lin_gray = train_model(X_train_flat, Y_train, "linear", lr=0.1,  iters=50, batch_size=512)
    params_h1_gray,  hist_h1_gray  = train_model(X_train_flat, Y_train, "mlp",    n_h1=128, lr=0.1, iters=50, batch_size=512)

    print("\nTest du MLP sur CIFAR-10 en couleur")
    # lr=0.01 pour 3072 features : lr=0.1 provoque des gradients trop grands
    params_lin_color, hist_lin_color = train_model(X_train_color_flat, Y_train, "linear", lr=0.01, iters=50, batch_size=512)
    params_h1_color,  hist_h1_color  = train_model(X_train_color_flat, Y_train, "mlp",   n_h1=128, lr=0.05, iters=50, batch_size=512)

    # ── Évaluation MLP / Linéaire ─────────────────────────────────
    pred_lin_gray,  acc_lin_gray  = evaluer_modele(X_test_flat,       y_test, params_lin_gray,  "linear")
    pred_h1_gray,   acc_h1_gray   = evaluer_modele(X_test_flat,       y_test, params_h1_gray,   "mlp")
    pred_lin_color, acc_lin_color = evaluer_modele(X_test_color_flat, y_test, params_lin_color, "linear")
    pred_h1_color,  acc_h1_color  = evaluer_modele(X_test_color_flat, y_test, params_h1_color,  "mlp")

    # ── Entraînement CNN ──────────────────────────────────────────
    import torch as _torch

    if os.path.exists(CNN_WEIGHTS_PATH):
        print(f"\n[LOAD] Modèle CNN existant — chargement depuis {CNN_WEIGHTS_PATH}")
        params_cnn = mc.init_cnn()
        params_cnn['model'].load_state_dict(
            _torch.load(CNN_WEIGHTS_PATH, map_location=params_cnn['device'])
        )
        params_cnn['model'].train(False)
        hist_train_cnn, hist_val_cnn = [], []
    else:
        params_cnn, hist_train_cnn, hist_val_cnn = train_model_cnn(
            X_train, y_train, lr=0.001, iters=40, batch_size=512
        )
        _torch.save(params_cnn['model'].state_dict(), CNN_WEIGHTS_PATH)
        print(f"[SAVE] Modèle CNN sauvegardé dans {CNN_WEIGHTS_PATH}")

    # ── Évaluation CNN ────────────────────────────────────────────
    _, acc_cnn_train = evaluer_modele(X_train, y_train, params_cnn, "cnn", ensemble="TRAIN")
    pred_cnn, acc_cnn_test = evaluer_modele(X_test, y_test, params_cnn, "cnn")

    # ── Tableau comparatif ────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Modèle':<28} {'Données':<10} {'Précision test':>14} {'Erreur test':>14}")
    print("-" * 72)
    print(f"{'Linéaire':<28} {'Gris':<10} {acc_lin_gray*100:>13.2f}% {(1-acc_lin_gray)*100:>13.2f}%")
    print(f"{'MLP H=1':<28} {'Gris':<10} {acc_h1_gray*100:>13.2f}% {(1-acc_h1_gray)*100:>13.2f}%")
    print(f"{'Linéaire':<28} {'Couleur':<10} {acc_lin_color*100:>13.2f}% {(1-acc_lin_color)*100:>13.2f}%")
    print(f"{'MLP H=1':<28} {'Couleur':<10} {acc_h1_color*100:>13.2f}% {(1-acc_h1_color)*100:>13.2f}%")
    print(f"{'CNN (PyTorch)':<28} {'Couleur':<10} {acc_cnn_test*100:>13.2f}% {(1-acc_cnn_test)*100:>13.2f}%")
    print("=" * 72)
    print("\nRéférence articles scientifiques :")
    print(f"  Conv. Deep Belief Networks (2010) :  78.9%    erreur: 21.1%")
    print(f"  Maxout Networks (2013)             :  90.6%    erreur:  9.4%")
    print(f"  ViT (2021)                         :  99.5%    erreur:  0.5%")

    # ── Visualisations ────────────────────────────────────────────
    visualiser_erreurs(
        X_test, y_test, pred_cnn,
        titre="Exemples mal classés par le CNN CIFAR-10",
        class_names=CIFAR_CLASSES,
        save_path=S("erreurs_cnn.png"),
    )

    tracer_convergence(
        [hist_lin_gray, hist_h1_gray, hist_lin_color, hist_h1_color, hist_train_cnn, hist_val_cnn],
        ["Linéaire (gris)", "MLP H=1 (gris)", "Linéaire (couleur)", "MLP H=1 (couleur)", "CNN (train)", "CNN (validation)"],
        titre="Convergence CIFAR-10 (Log Loss)",
        save_path=S("convergence.png"),
    )

    acp = PCA(n_components=2)
    acp.fit(X_train_gray.reshape(X_train_gray.shape[0], -1))
    X_test_2d = acp.transform(X_test_gray.reshape(X_test_gray.shape[0], -1))

    tracer_acp(
        X_test_2d, y_test,
        titre="ACP CIFAR-10 (niveaux de gris)",
        class_names=CIFAR_CLASSES,
        save_path=S("acp.png"),
    )

finally:
    log.close()
