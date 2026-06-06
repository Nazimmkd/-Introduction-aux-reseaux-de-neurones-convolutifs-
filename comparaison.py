"""
comparaison.py — Étude comparative des fonctions d'activation et des architectures MLP
Datasets : MNIST | CIFAR-10 gris | CIFAR-10 couleur | CNN (référence)
Sorties   : sorties/comparaison/
"""
import numpy as np
import os

import utils.sorties as sorties
from utils.donnees import charger_mnist, charger_cifar10
from utils.preprocessing import vector_label, rgb_to_grayscale
from utils.entrainement import train_model, train_model_cnn
from utils.evaluation import evaluer_modele
from utils.gpu import to_gpu, liberer_gpu
from utils.visualisation import (
    tracer_convergence,
    tracer_comparaison_barres,
    tracer_heatmap_resultats,
)
import modeles.modele_convolutif as mc

C = lambda nom: sorties.chemin("comparaison", nom)
M = lambda nom: sorties.chemin("modeles",     nom)

CNN_WEIGHTS_PATH = M("cnn_cifar10.pth")


# ══════════════════════════════════════════════════════════════════
#  GRILLE D'EXPÉRIENCES
# ══════════════════════════════════════════════════════════════════

ACTIVATIONS = ["sigmoid", "tanh", "relu", "heaviside"]

ARCHITECTURES = {
    "1×[64]":           [64],
    "1×[128]":          [128],
    "2×[128,64]":       [128, 64],
    "3×[128,64,32]":    [128, 64, 32],
    "3×[256,128,64]":   [256, 128, 64],
}

CONFIGS = {
    "mnist":         {"lr": 0.1,  "iters": 50, "batch_size": 256},
    # CIFAR gris (1024 features) — même LR que cifar.py
    "cifar_gris":    {"lr": 0.1,  "iters": 50, "batch_size": 512},
    # CIFAR couleur (3072 features) — LR réduit comme dans cifar.py (gradients plus grands)
    "cifar_couleur": {"lr": 0.05, "iters": 50, "batch_size": 512},
}

LR_OVERRIDES = {}   # ex: {"relu": 0.05} — décommenter pour ajuster par activation


# ══════════════════════════════════════════════════════════════════
#  MOTEUR D'EXPÉRIMENTATION MLP
# ══════════════════════════════════════════════════════════════════

def lancer_experience(X_train, Y_train, X_test, y_test, config, label):
    """
    Entraîne un MLP pour chaque (activation × architecture).
    Retourne :
      resultats   : {(activation, arch_nom): accuracy}
      historiques : {(activation, arch_nom): [loss par époque]}
    """
    resultats   = {}
    historiques = {}

    for activation in ACTIVATIONS:
        lr = LR_OVERRIDES.get(activation, config["lr"])
        for arch_nom, couches in ARCHITECTURES.items():
            cle = (activation, arch_nom)
            print(f"\n[{label}] activation={activation:>10} | arch={arch_nom}")
            params, history = train_model(
                X_train, Y_train, "mlp",
                couches_cachees=couches,
                activation=activation,
                lr=lr,
                iters=config["iters"],
                batch_size=config["batch_size"],
            )
            _, acc = evaluer_modele(X_test, y_test, params, "mlp", activation=activation)
            resultats[cle]   = acc
            historiques[cle] = history

    return resultats, historiques


# ══════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def afficher_tableau(resultats, titre):
    """Tableau texte : lignes = architectures (triées), colonnes = activations."""
    col_w  = 11
    arch_w = 22

    print(f"\n  {titre} — Précision test (%)")
    print("=" * (arch_w + col_w * len(ACTIVATIONS) + col_w + 2))
    entete = f"{'Architecture':<{arch_w}}" + "".join(f"{a:>{col_w}}" for a in ACTIVATIONS) + f"{'  Meilleur':>{col_w}}"
    print(entete)
    print("-" * (arch_w + col_w * len(ACTIVATIONS) + col_w + 2))

    def meilleure(arch):
        return max(resultats.get((a, arch), 0) for a in ACTIVATIONS)

    for arch in sorted(ARCHITECTURES, key=meilleure, reverse=True):
        valeurs  = [resultats.get((a, arch), 0) * 100 for a in ACTIVATIONS]
        idx_max  = valeurs.index(max(valeurs))
        cellules = ""
        for i, v in enumerate(valeurs):
            marqueur = " *" if i == idx_max else "  "
            cellules += f"{v:>{col_w - 2}.1f}{marqueur}"
        print(f"{arch:<{arch_w}}{cellules}  {max(valeurs):>8.1f}%")

    print("-" * (arch_w + col_w * len(ACTIVATIONS) + col_w + 2))
    maxs = [max(resultats.get((a, arch), 0) * 100 for arch in ARCHITECTURES) for a in ACTIVATIONS]
    print(f"{'MAX (toutes arch.)':<{arch_w}}" + "".join(f"{v:>{col_w}.1f}" for v in maxs))
    print("=" * (arch_w + col_w * len(ACTIVATIONS) + col_w + 2))


def afficher_baselines_cifar(acc_lin_gray, acc_lin_color, acc_cnn):
    """Résumé des modèles non-MLP comme ligne de référence."""
    print(f"\n  CIFAR-10 — Modeles de reference (sans grille activation)")
    print("=" * 55)
    print(f"{'Modele':<30} {'Donnees':<10} {'Precision':>12}")
    print("-" * 55)
    print(f"{'Lineaire':<30} {'Gris':<10} {acc_lin_gray  * 100:>11.2f}%")
    print(f"{'Lineaire':<30} {'Couleur':<10} {acc_lin_color * 100:>11.2f}%")
    print(f"{'CNN (PyTorch)':<30} {'Couleur':<10} {acc_cnn       * 100:>11.2f}%")
    print("=" * 55)
    print("  Reference articles scientifiques :")
    print("    Conv. Deep Belief Networks (2010) :  78.9%")
    print("    Maxout Networks (2013)             :  90.6%")
    print("    ViT (2021)                         :  99.5%")


def sauvegarder_convergences(historiques, prefixe, titre_prefixe):
    """Une courbe de convergence par architecture, toutes activations superposées."""
    for arch_nom in ARCHITECTURES:
        nom_f = arch_nom.replace("×", "x").replace("[", "").replace("]", "").replace(",", "-")
        tracer_convergence(
            histories=[historiques[(act, arch_nom)] for act in ACTIVATIONS],
            labels=ACTIVATIONS,
            titre=f"{titre_prefixe} — Convergence {arch_nom}",
            save_path=C(f"{prefixe}_convergence_{nom_f}.png"),
        )


# ══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

log = sorties.demarrer_log("comparaison")
try:

    # ── MNIST ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARAISON ACTIVATIONS x ARCHITECTURES — MNIST")
    print("=" * 60)
    X_m, y_m, Xt_m, yt_m = charger_mnist()
    Y_m = to_gpu(vector_label(y_m))
    X_m = to_gpu(X_m)

    res_mnist, hist_mnist = lancer_experience(X_m, Y_m, Xt_m, yt_m, CONFIGS["mnist"], "MNIST")
    afficher_tableau(res_mnist, "MNIST")
    tracer_comparaison_barres(
        res_mnist, ACTIVATIONS, ARCHITECTURES,
        titre="MNIST — Precision par activation et architecture",
        save_path=C("mnist_barres.png"),
    )
    tracer_heatmap_resultats(
        res_mnist, ACTIVATIONS, list(ARCHITECTURES.keys()),
        titre="MNIST — Heatmap precision (%)",
        save_path=C("mnist_heatmap.png"),
    )
    sauvegarder_convergences(hist_mnist, "mnist", "MNIST")
    del X_m, Y_m
    liberer_gpu()

    # ── CIFAR-10 — chargement et préparation ──────────────────────
    print("\n" + "=" * 60)
    print("CHARGEMENT CIFAR-10")
    print("=" * 60)
    X_c, y_c, Xt_c, yt_c = charger_cifar10()

    # ── CIFAR : section gris (GPU gris uniquement) ────────────────
    print("\n--- Baseline lineaire (gris) ---")
    X_train_gray = rgb_to_grayscale(X_c)
    X_test_gray  = rgb_to_grayscale(Xt_c)
    X_train_flat_gray = to_gpu(X_train_gray.reshape(X_train_gray.shape[0], -1).T)
    X_test_flat_gray  = to_gpu(X_test_gray.reshape(X_test_gray.shape[0], -1).T)
    Y_train_c         = to_gpu(vector_label(y_c))
    del X_train_gray, X_test_gray

    params_lin_gray, hist_lin_gray = train_model(
        X_train_flat_gray, Y_train_c, "linear", lr=0.1, iters=50, batch_size=512)
    _, acc_lin_gray = evaluer_modele(X_test_flat_gray, yt_c, params_lin_gray, "linear")
    del params_lin_gray

    print("\n" + "=" * 60)
    print("COMPARAISON ACTIVATIONS x ARCHITECTURES — CIFAR gris")
    print("=" * 60)
    res_gris, hist_gris = lancer_experience(
        X_train_flat_gray, Y_train_c, X_test_flat_gray, yt_c,
        CONFIGS["cifar_gris"], "CIFAR gris",
    )
    afficher_tableau(res_gris, "CIFAR gris")
    del X_train_flat_gray, X_test_flat_gray, Y_train_c
    liberer_gpu()

    # ── CIFAR : section couleur (GPU couleur uniquement) ──────────
    print("\n--- Baseline lineaire (couleur) ---")
    X_train_flat_color = to_gpu(X_c.reshape(X_c.shape[0], -1).T)
    X_test_flat_color  = to_gpu(Xt_c.reshape(Xt_c.shape[0], -1).T)
    Y_train_c          = to_gpu(vector_label(y_c))

    params_lin_color, hist_lin_color = train_model(
        X_train_flat_color, Y_train_c, "linear", lr=0.01, iters=50, batch_size=512)
    _, acc_lin_color = evaluer_modele(X_test_flat_color, yt_c, params_lin_color, "linear")
    del params_lin_color

    print("\n" + "=" * 60)
    print("COMPARAISON ACTIVATIONS x ARCHITECTURES — CIFAR couleur")
    print("=" * 60)
    res_color, hist_color = lancer_experience(
        X_train_flat_color, Y_train_c, X_test_flat_color, yt_c,
        CONFIGS["cifar_couleur"], "CIFAR couleur",
    )
    afficher_tableau(res_color, "CIFAR couleur")
    del X_train_flat_color, X_test_flat_color, Y_train_c
    liberer_gpu()

    # ── CIFAR : CNN ───────────────────────────────────────────────
    print("\n--- CNN PyTorch ---")
    import torch as _torch
    if os.path.exists(CNN_WEIGHTS_PATH):
        print(f"[LOAD] CNN existant — {CNN_WEIGHTS_PATH}")
        params_cnn = mc.init_cnn()
        params_cnn['model'].load_state_dict(
            _torch.load(CNN_WEIGHTS_PATH, map_location=params_cnn['device'])
        )
        params_cnn['model'].train(False)
        hist_train_cnn, hist_val_cnn = [], []
    else:
        params_cnn, hist_train_cnn, hist_val_cnn = train_model_cnn(
            X_c, y_c, lr=0.001, iters=40, batch_size=512
        )
        _torch.save(params_cnn['model'].state_dict(), CNN_WEIGHTS_PATH)
        print(f"[SAVE] CNN sauvegarde dans {CNN_WEIGHTS_PATH}")
    _, acc_cnn = evaluer_modele(Xt_c, yt_c, params_cnn, "cnn")

    # ── CIFAR : résumé final ──────────────────────────────────────
    afficher_baselines_cifar(acc_lin_gray, acc_lin_color, acc_cnn)

    # ── Visualisations CIFAR ──────────────────────────────────────
    tracer_comparaison_barres(
        res_gris, ACTIVATIONS, ARCHITECTURES,
        titre="CIFAR gris — Precision par activation et architecture",
        save_path=C("cifar_gris_barres.png"),
    )
    tracer_heatmap_resultats(
        res_gris, ACTIVATIONS, list(ARCHITECTURES.keys()),
        titre="CIFAR gris — Heatmap precision (%)",
        save_path=C("cifar_gris_heatmap.png"),
    )
    tracer_comparaison_barres(
        res_color, ACTIVATIONS, ARCHITECTURES,
        titre="CIFAR couleur — Precision par activation et architecture",
        save_path=C("cifar_couleur_barres.png"),
    )
    tracer_heatmap_resultats(
        res_color, ACTIVATIONS, list(ARCHITECTURES.keys()),
        titre="CIFAR couleur — Heatmap precision (%)",
        save_path=C("cifar_couleur_heatmap.png"),
    )
    # Courbe de convergence CNN sur le même graphe que les MLP pour comparaison
    tracer_convergence(
        [hist_lin_gray, hist_lin_color, hist_train_cnn, hist_val_cnn],
        ["Lineaire (gris)", "Lineaire (couleur)", "CNN (train)", "CNN (validation)"],
        titre="CIFAR-10 — Baselines convergence",
        save_path=C("cifar_baselines_convergence.png"),
    )
    sauvegarder_convergences(hist_gris,  "cifar_gris",    "CIFAR gris")
    sauvegarder_convergences(hist_color, "cifar_couleur", "CIFAR couleur")

finally:
    log.close()
