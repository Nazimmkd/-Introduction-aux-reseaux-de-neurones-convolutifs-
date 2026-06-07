#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py -- Script principal SM604
Projet : De la classification des chiffres manuscrits a la detection de cancers du sein
Execution unique produisant toutes les figures et metriques du rapport LaTeX.
"""

# ================================================================
# === IMPORTS ET CONFIGURATION ===
# ================================================================

import os, sys, random, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc as sklearn_auc
import torch

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "sorties", "figures")
MODELES_DIR = os.path.join(BASE_DIR, "sorties", "modeles")
LOG_PATH    = os.path.join(BASE_DIR, "sorties", "execution.txt")
for _d in (FIGURES_DIR, MODELES_DIR, os.path.dirname(LOG_PATH)):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from utils.preprocessing import vector_label, rgb_to_grayscale
from utils.gpu           import to_gpu, to_cpu, liberer_gpu, GPU_AVAILABLE
from utils.donnees       import charger_mnist, charger_cifar10
from utils.donnees_ddsm  import charger_cbis_ddsm
from utils.entrainement  import (train_model, train_model_cnn,
                                  train_model_cnn_binaire, train_ablation_cnn)
from utils.evaluation    import evaluer_modele, obtenir_scores, calibrer_seuil, matrice_confusion
from utils.visualisation import (tracer_convergence, tracer_acp,
                                  tracer_confusion, CIFAR_CLASSES)
import modeles.modele_convolutif as mc

# ================================================================
# === SEEDS ET DEVICE ===
# ================================================================

SEED = 42

def fixer_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

fixer_seeds()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})

COULEURS = {"bleu": "#2196F3", "orange": "#FF9800", "vert": "#4CAF50",
            "rouge": "#F44336", "violet": "#9C27B0", "gris": "#607D8B",
            "jaune": "#FFC107", "cyan": "#00BCD4"}
PALETTE = list(COULEURS.values())


class _Tee:
    def __init__(self, path):
        self._term = sys.__stdout__
        self._file = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self._term.write(msg); self._file.write(msg); self._file.flush()

    def flush(self):
        self._term.flush(); self._file.flush()

    def close(self):
        sys.stdout = self._term; self._file.close()


_log = _Tee(LOG_PATH)
sys.stdout = _log
print(f"Device : {DEVICE} | CuPy GPU : {GPU_AVAILABLE}")
print(f"Figures : {FIGURES_DIR}")

# ─── Utilitaires ───────────────────────────────────────────────

def fig(name): return os.path.join(FIGURES_DIR, name)
def mdl(name): return os.path.join(MODELES_DIR, name)


def sauver_pt(path, **kwargs):
    """Sauvegarde un dict d'objets Python via torch.save."""
    torch.save(kwargs, path)


def charger_pt(path):
    """Charge un dict sauvegarde via sauver_pt."""
    return torch.load(path, weights_only=False)


def compter_params(couches):
    return sum(couches[i] * couches[i+1] + couches[i+1] for i in range(len(couches) - 1))


def fmt_n(n): return f"{n:,}".replace(",", " ")


# ReLU produit des gradients non bornés (dérivée = 0 ou 1) → SGD instable à LR élevé.
# Ce facteur réduit le LR d'un ordre de grandeur par rapport à sigmoid/tanh.
LR_RELU_FACTOR = 0.1

def _lr_act(base_lr, activation):
    """Retourne base_lr × LR_RELU_FACTOR si activation='relu', sinon base_lr."""
    return base_lr * LR_RELU_FACTOR if activation == "relu" else base_lr


def metriques_cm(cm):
    """Retourne (acc, sens, spec, FN, FP) depuis une matrice de confusion 2x2."""
    TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    tot = cm.sum()
    acc  = (TP + TN) / tot if tot > 0 else 0.
    sens = TP / (TP + FN)  if (TP + FN) > 0 else 0.
    spec = TN / (TN + FP)  if (TN + FP) > 0 else 0.
    return acc, sens, spec, FN, FP


def detecter_trivial(sens, spec):
    if sens > 0.95 and spec < 0.10: return "Predit tout Malin"
    if spec > 0.95 and sens < 0.10: return "Predit tout Benin"
    return "Equilibre"


def tracer_double_confusion(cm_05, cm_cal, seuil_cal, titre, save_path):
    """2 matrices de confusion cote a cote : seuil 0.50 / seuil calibre."""
    fl, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    cls = ["Benin", "Malin"]
    for ax, cm, st in zip(axes,
                          [cm_05, cm_cal],
                          ["Seuil = 0.50", f"Seuil calibre = {seuil_cal:.2f}"]):
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([0, 1]); ax.set_xticklabels(cls)
        ax.set_yticks([0, 1]); ax.set_yticklabels(cls)
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel"); ax.set_title(st, fontsize=12)
        tot = cm.sum()
        for i in range(2):
            for j in range(2):
                v = int(cm[i, j])
                col = "white" if v > cm.max() / 1.8 else "black"
                ax.text(j, i, f"{v}\n({v / tot * 100:.1f}%)",
                        ha="center", va="center", color=col, fontsize=11, fontweight="bold")
    fl.suptitle(titre, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def tracer_heatmap_custom(data, act_lbl, arch_lbl, titre, save_path):
    """Heatmap acc_test rouge->vert avec valeurs annotees."""
    fl, ax = plt.subplots(figsize=(10, 5))
    cmap = LinearSegmentedColormap.from_list("rg", ["#F44336", "#FFC107", "#4CAF50"])
    vmin, vmax = data.min(), data.max()
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(arch_lbl))); ax.set_xticklabels(arch_lbl, rotation=30, ha="right")
    ax.set_yticks(range(len(act_lbl)));  ax.set_yticklabels(act_lbl)
    mid = (vmin + vmax) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            col = "black" if data[i, j] > mid else "white"
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", fontsize=9, color=col)
    plt.colorbar(im, ax=ax, label="Accuracy test (%)")
    ax.set_title(titre); plt.tight_layout(); plt.savefig(save_path); plt.close()


ACTIVATIONS_GRILLE = ["sigmoid", "tanh", "relu", "heaviside"]
ARCHS_GRILLE = {
    "[64]":         [64],
    "[128]":        [128],
    "[128,64]":     [128, 64],
    "[128,64,32]":  [128, 64, 32],
    "[256,128,64]": [256, 128, 64],
}

# ================================================================
# === PARTIE 1 : MNIST ===
# ================================================================

print("\n" + "=" * 64)
print("PARTIE 1 -- MNIST")
print("=" * 64)

try:
    # -- S1.1 Chargement ------------------------------------------------
    print("\n[1.1] Chargement MNIST...")
    X_tr_mn, y_tr_mn, X_te_mn, y_te_mn = charger_mnist()
    Y_tr_mn = vector_label(y_tr_mn, 10)
    print(f"  Train {X_tr_mn.shape[1]} | Test {X_te_mn.shape[1]}")
    mn_res  = {}
    mn_pred = {}

    # -- S1.2 Lineaire --------------------------------------------------
    print("\n[1.2] Modele lineaire MNIST...")
    _path = mdl("mnist_lineaire.pt")
    fixer_seeds()
    if os.path.exists(_path):
        _d = charger_pt(_path); params_lin_mn, hist_lin_mn = _d["params"], _d["hist"]
    else:
        params_lin_mn, hist_lin_mn = train_model(
            X_tr_mn, Y_tr_mn, "linear", lr=0.1, iters=100, batch_size=256)
        sauver_pt(_path, params=params_lin_mn, hist=hist_lin_mn)
    fixer_seeds()
    _, _at = evaluer_modele(X_tr_mn, y_tr_mn, params_lin_mn, "linear")
    mn_pred["Lineaire"], _ae = evaluer_modele(X_te_mn, y_te_mn, params_lin_mn, "linear")
    mn_res["Lineaire"] = {
        "params": 784 * 10 + 10, "hist": hist_lin_mn, "couches": [784, 10],
        "acc_train": _at, "acc_test": _ae, "err_train": 1 - _at, "err_test": 1 - _ae,
    }
    print(f"  Lineaire | {784*10+10} params | acc_test={_ae*100:.2f}%")

    # -- S1.3 MLP H=1 ---------------------------------------------------
    print("\n[1.3] MLP H=1 -- 4 tailles...")
    _configs_h1 = [64, 128, 256, 512]
    for _h in _configs_h1:
        _cle  = f"MLP H=1 [{_h}]"
        _path = mdl(f"mnist_mlp_h1_{_h}.pt")
        fixer_seeds()
        if os.path.exists(_path):
            _d = charger_pt(_path); _p, _hist = _d["params"], _d["hist"]
        else:
            _p, _hist = train_model(
                X_tr_mn, Y_tr_mn, "mlp",
                couches_cachees=[_h], activation="relu",
                lr=_lr_act(0.1, "relu"), iters=100, batch_size=256)
            sauver_pt(_path, params=_p, hist=_hist)
        fixer_seeds()
        _, _at = evaluer_modele(X_tr_mn, y_tr_mn, _p, "mlp", activation="relu")
        mn_pred[_cle], _ae = evaluer_modele(X_te_mn, y_te_mn, _p, "mlp", activation="relu")
        _cc = [784, _h, 10]
        mn_res[_cle] = {
            "params": compter_params(_cc), "hist": _hist, "couches": _cc,
            "acc_train": _at, "acc_test": _ae, "err_train": 1 - _at, "err_test": 1 - _ae,
        }
        print(f"  {_cle} | {compter_params(_cc)} params | acc_test={_ae*100:.2f}%")

    # -- S1.4 MLP H=2 ---------------------------------------------------
    print("\n[1.4] MLP H=2 -- 3 configurations...")
    _configs_h2 = [[128, 64], [256, 128], [512, 256]]
    for _hl in _configs_h2:
        _cle  = f"MLP H=2 [{','.join(map(str,_hl))}]"
        _path = mdl(f"mnist_mlp_h2_{'_'.join(map(str,_hl))}.pt")
        fixer_seeds()
        if os.path.exists(_path):
            _d = charger_pt(_path); _p, _hist = _d["params"], _d["hist"]
        else:
            _p, _hist = train_model(
                X_tr_mn, Y_tr_mn, "mlp",
                couches_cachees=_hl, activation="relu",
                lr=_lr_act(0.1, "relu"), iters=100, batch_size=256)
            sauver_pt(_path, params=_p, hist=_hist)
        fixer_seeds()
        _, _at = evaluer_modele(X_tr_mn, y_tr_mn, _p, "mlp", activation="relu")
        mn_pred[_cle], _ae = evaluer_modele(X_te_mn, y_te_mn, _p, "mlp", activation="relu")
        _cc = [784] + _hl + [10]
        mn_res[_cle] = {
            "params": compter_params(_cc), "hist": _hist, "couches": _cc,
            "acc_train": _at, "acc_test": _ae, "err_train": 1 - _at, "err_test": 1 - _ae,
        }
        print(f"  {_cle} | {compter_params(_cc)} params | acc_test={_ae*100:.2f}%")

    # -- S1.5 Grille activation x architecture -------------------------
    print("\n[1.5] Grille activation x architecture MNIST (50 iter)...")
    _path_g = mdl("mnist_grille_act.pt")
    if os.path.exists(_path_g):
        mn_grille = charger_pt(_path_g)["grille"]
    else:
        mn_grille = {}
        for _act in ACTIVATIONS_GRILLE:
            for _anm, _arch in ARCHS_GRILLE.items():
                fixer_seeds()
                try:
                    _pg, _ = train_model(X_tr_mn, Y_tr_mn, "mlp",
                                         couches_cachees=_arch, activation=_act,
                                         lr=_lr_act(0.1, _act), iters=50, batch_size=256)
                    _, _acc = evaluer_modele(X_te_mn, y_te_mn, _pg, "mlp", activation=_act)
                    mn_grille[(_act, _anm)] = _acc
                    print(f"    {_act:10s} x {_anm:15s} -> {_acc*100:.2f}%")
                except Exception as _e:
                    print(f"    [ERREUR] {_act} x {_anm} : {_e}")
                    mn_grille[(_act, _anm)] = 0.
        sauver_pt(_path_g, grille=mn_grille)

    # -- S1.6 Figures MNIST --------------------------------------------
    print("\n[1.6] Figures MNIST...")

    try:
        fl, ax = plt.subplots(figsize=(10, 6))
        for _lbl, _cle, _col in [
            ("Lineaire",          "Lineaire",           COULEURS["bleu"]),
            ("MLP H=1 [256]",     "MLP H=1 [256]",      COULEURS["orange"]),
            ("MLP H=2 [256,128]", "MLP H=2 [256,128]",  COULEURS["vert"]),
        ]:
            _h = mn_res.get(_cle, {}).get("hist", [])
            if _h:
                ax.semilogy(range(1, len(_h)+1), _h, label=_lbl, color=_col, linewidth=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Log-perte (echelle log)")
        ax.set_title("Convergence des modeles MNIST"); ax.legend(loc="upper right")
        plt.tight_layout(); plt.savefig(fig("fig_mnist_convergence.pdf")); plt.close()
        print("  fig_mnist_convergence.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_mnist_convergence : {_e}")

    try:
        _lb, _ab, _pb, _cb = [], [], [], []
        for _h in _configs_h1:
            _cle = f"MLP H=1 [{_h}]"
            _lb.append(f"H=1\n[{_h}]"); _ab.append(mn_res[_cle]["acc_test"] * 100)
            _pb.append(mn_res[_cle]["params"]); _cb.append(COULEURS["bleu"])
        for _hl in _configs_h2:
            _cle = f"MLP H=2 [{','.join(map(str,_hl))}]"
            _lb.append(f"H=2\n[{','.join(map(str,_hl))}]")
            _ab.append(mn_res[_cle]["acc_test"] * 100)
            _pb.append(mn_res[_cle]["params"]); _cb.append(COULEURS["orange"])
        fl, ax = plt.subplots(figsize=(12, 6))
        _x = np.arange(len(_lb))
        _bars = ax.bar(_x, _ab, color=_cb, alpha=0.85, edgecolor="white")
        for _bar, _np in zip(_bars, _pb):
            ax.text(_bar.get_x() + _bar.get_width() / 2., _bar.get_height() + 0.05,
                    fmt_n(_np), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(_x); ax.set_xticklabels(_lb, fontsize=9)
        ax.set_ylabel("Accuracy test (%)")
        ax.set_title("Effet de la taille des couches cachees -- MNIST")
        ax.set_ylim(max(85, min(_ab) - 1), 100)
        ax.legend(handles=[
            Patch(facecolor=COULEURS["bleu"],   label="H=1 (1 couche cachee)"),
            Patch(facecolor=COULEURS["orange"],  label="H=2 (2 couches cachees)"),
        ])
        plt.tight_layout(); plt.savefig(fig("fig_mnist_tailles_couches.pdf")); plt.close()
        print("  fig_mnist_tailles_couches.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_mnist_tailles_couches : {_e}")

    try:
        _rng = np.random.RandomState(SEED)
        _idx = np.concatenate(
            [_rng.choice(np.where(y_te_mn == c)[0], 200, replace=False) for c in range(10)])
        _X2d = PCA(n_components=2, random_state=SEED).fit_transform(X_te_mn[:, _idx].T)
        _y   = y_te_mn[_idx]
        fl, ax = plt.subplots(figsize=(10, 8))
        for _c, _col in zip(range(10), PALETTE):
            _m = _y == _c
            ax.scatter(_X2d[_m, 0], _X2d[_m, 1], c=_col, label=str(_c), s=15, alpha=0.6)
        for _paire in [(3, 8), (4, 9)]:
            for _c in _paire:
                _m = _y == _c
                if _m.any():
                    ax.annotate(f"cl.{_c}",
                                (_X2d[_m, 0].mean(), _X2d[_m, 1].mean()),
                                fontsize=9, fontweight="bold")
        ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
        ax.set_title("ACP 2D -- MNIST (2000 exemples de test, stratifie)")
        ax.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.tight_layout(); plt.savefig(fig("fig_mnist_acp.pdf")); plt.close()
        print("  fig_mnist_acp.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_mnist_acp : {_e}")

    _n_erreurs_communes = 0
    try:
        _plin = mn_pred.get("Lineaire",           np.zeros(len(y_te_mn), int))
        _ph1  = mn_pred.get("MLP H=1 [256]",      np.zeros(len(y_te_mn), int))
        _ph2  = mn_pred.get("MLP H=2 [256,128]",  np.zeros(len(y_te_mn), int))
        _etous = np.where((_plin != y_te_mn) & (_ph1 != y_te_mn) & (_ph2 != y_te_mn))[0]
        _n_erreurs_communes = len(_etous)
        if _n_erreurs_communes > 0:
            fl, axes = plt.subplots(2, 10, figsize=(20, 5))
            for _i, _idx in enumerate(_etous[:20]):
                _r, _ci = divmod(_i, 10)
                axes[_r, _ci].imshow(X_te_mn[:, _idx].reshape(28, 28), cmap="gray")
                axes[_r, _ci].axis("off")
                axes[_r, _ci].set_title(
                    f"V:{y_te_mn[_idx]}\nL:{_plin[_idx]}/M1:{_ph1[_idx]}/M2:{_ph2[_idx]}",
                    fontsize=6)
            plt.suptitle("Exemples mal classes par les 3 modeles simultanement", fontsize=11)
            plt.tight_layout(); plt.savefig(fig("fig_mnist_erreurs_communes.pdf")); plt.close()
        print(f"  fig_mnist_erreurs_communes.pdf OK ({_n_erreurs_communes} exemples)")
    except Exception as _e:
        print(f"  [ERREUR] fig_mnist_erreurs_communes : {_e}")

    try:
        _hmap = np.array([
            [mn_grille.get((_a, _ar), 0.) * 100 for _ar in ARCHS_GRILLE]
            for _a in ACTIVATIONS_GRILLE
        ])
        tracer_heatmap_custom(
            _hmap, ACTIVATIONS_GRILLE, list(ARCHS_GRILLE.keys()),
            "Heatmap Activation x Architecture -- MNIST (acc. test %)",
            fig("fig_mnist_heatmap_activations.pdf"))
        print("  fig_mnist_heatmap_activations.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_mnist_heatmap_activations : {_e}")

    # -- S1.7 Trace MNIST -----------------------------------------------
    print("\n[1.7] Trace recapitulative MNIST")
    _S = "=" * 64; _L = "-" * 64
    print(f"\n{_S}"); print("MNIST -- Resultats"); print(_S)
    print(f"{'Modele':<24} {'Params':>9}  {'Err.Train':>9}  {'Err.Test':>8}  {'Acc.Test':>8}")
    print(_L)
    for _cle in (["Lineaire"] +
                 [f"MLP H=1 [{_h}]" for _h in _configs_h1] +
                 [f"MLP H=2 [{','.join(map(str,_hl))}]" for _hl in _configs_h2]):
        _r = mn_res.get(_cle)
        if _r:
            print(f"{_cle:<24} {fmt_n(_r['params']):>9}  "
                  f"{_r['err_train']*100:>8.2f}%  "
                  f"{_r['err_test']*100:>7.2f}%  "
                  f"{_r['acc_test']*100:>7.2f}%")
    print(_S)
    print(f"Erreurs communes aux 3 modeles principaux : {_n_erreurs_communes} exemples")

except Exception as _e_p1:
    print(f"\n[ERREUR] Partie 1 MNIST : {_e_p1}"); traceback.print_exc()

# ================================================================
# === PARTIE 2 : CIFAR-10 ===
# ================================================================

print("\n" + "=" * 64)
print("PARTIE 2 -- CIFAR-10")
print("=" * 64)

try:
    # -- S2.1 Chargement ------------------------------------------------
    print("\n[2.1] Chargement CIFAR-10...")
    X_tr_cf, y_tr_cf, X_te_cf, y_te_cf = charger_cifar10()
    X_tr_gray = rgb_to_grayscale(X_tr_cf)
    X_te_gray = rgb_to_grayscale(X_te_cf)
    X_tr_gf = to_gpu(X_tr_gray.reshape(len(X_tr_gray), -1).T.astype(np.float32))
    X_te_gf = X_te_gray.reshape(len(X_te_gray), -1).T.astype(np.float32)
    X_tr_cf2 = to_gpu(X_tr_cf.reshape(len(X_tr_cf), -1).T.astype(np.float32))
    X_te_cf2 = X_te_cf.reshape(len(X_te_cf), -1).T.astype(np.float32)
    Y_tr_cf  = to_gpu(vector_label(y_tr_cf, 10))
    print(f"  Train {X_tr_cf.shape[0]} | Test {X_te_cf.shape[0]}")
    cf_res = {}

    # -- S2.2 MLP / Lineaire --------------------------------------------
    print("\n[2.2] Modeles MLP/Lineaire CIFAR-10...")
    _cfgs_mlp = [
        ("Lin.Gris",            "linear", [],        "relu",    "Gris",    X_tr_gf,  X_te_gf,  0.10,  512),
        ("MLP128relu.Gris",     "mlp",    [128],     "relu",    "Gris",    X_tr_gf,  X_te_gf,  0.01,  512),
        ("MLP256relu.Gris",     "mlp",    [256],     "relu",    "Gris",    X_tr_gf,  X_te_gf,  0.01,  512),
        ("Lin.Coul",            "linear", [],        "relu",    "Couleur", X_tr_cf2, X_te_cf2,  0.01,  512),
        ("MLP128relu.Coul",     "mlp",    [128],     "relu",    "Couleur", X_tr_cf2, X_te_cf2,  0.005, 512),
        ("MLP256relu.Coul",     "mlp",    [256],     "relu",    "Couleur", X_tr_cf2, X_te_cf2,  0.005, 512),
        ("MLP256-128relu.Coul", "mlp",    [256, 128],"relu",    "Couleur", X_tr_cf2, X_te_cf2,  0.001, 512),
    ]
    for _slug, _type, _cc, _act, _don, _Xtr, _Xte, _lr, _bs in _cfgs_mlp:
        _path = mdl(f"cifar_{_slug}.pt")
        fixer_seeds()
        if os.path.exists(_path):
            _d = charger_pt(_path); _p, _hist = _d["params"], _d["hist"]
        else:
            if _type == "linear":
                _p, _hist = train_model(_Xtr, Y_tr_cf, "linear", lr=_lr, iters=50, batch_size=_bs)
            else:
                _p, _hist = train_model(_Xtr, Y_tr_cf, "mlp",
                                        couches_cachees=_cc, activation=_act,
                                        lr=_lr, iters=50, batch_size=_bs)
            sauver_pt(_path, params=_p, hist=_hist)
        fixer_seeds()
        _, _at = evaluer_modele(_Xtr, y_tr_cf, _p, _type, activation=_act)
        _, _ae = evaluer_modele(_Xte, y_te_cf, _p, _type, activation=_act)
        cf_res[_slug] = {"acc_train": _at, "acc_test": _ae, "err_test": 1 - _ae,
                          "donnees": _don, "hist": _hist}
        print(f"  {_slug:<25} | acc_test={_ae*100:.2f}%")

    liberer_gpu()

    # -- S2.3 Filtres K1-K6 --------------------------------------------
    print("\n[2.3] Filtres K1-K6...")
    try:
        _idx_chat = np.where(y_te_cf == 3)[0][0]
        mc.visualiser_filtres_K(X_te_gray[_idx_chat], save_path=fig("fig_cifar_filtres.pdf"))
        print("  fig_cifar_filtres.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] filtres K1-K6 : {_e}")

    # -- S2.4 CNN PyTorch -----------------------------------------------
    print("\n[2.4] CNN PyTorch CIFAR-10...")
    _path_cnn_cf = mdl("cnn_cifar10.pth")
    fixer_seeds()
    if os.path.exists(_path_cnn_cf):
        print(f"  Chargement depuis {_path_cnn_cf}")
        params_cnn_cf = mc.init_cnn()
        params_cnn_cf["model"].load_state_dict(
            torch.load(_path_cnn_cf, map_location=params_cnn_cf["device"], weights_only=True))
        params_cnn_cf["model"].train(False)
        hist_tr_cnn_cf, hist_val_cnn_cf = [], []
    else:
        params_cnn_cf, hist_tr_cnn_cf, hist_val_cnn_cf = train_model_cnn(
            X_tr_cf, y_tr_cf, lr=0.001, iters=40, batch_size=512, patience=6)
        torch.save(params_cnn_cf["model"].state_dict(), _path_cnn_cf)
        print(f"  Sauvegarde dans {_path_cnn_cf}")
    fixer_seeds()
    _, _at_cnn = evaluer_modele(X_tr_cf, y_tr_cf, params_cnn_cf, "cnn", ensemble="TRAIN")
    _, _ae_cnn = evaluer_modele(X_te_cf, y_te_cf, params_cnn_cf, "cnn")
    cf_res["CNN4.Coul"] = {
        "acc_train": _at_cnn, "acc_test": _ae_cnn, "err_test": 1 - _ae_cnn, "donnees": "Couleur",
        "hist_train": hist_tr_cnn_cf, "hist_val": hist_val_cnn_cf,
    }
    print(f"  CNN | acc_train={_at_cnn*100:.2f}% | acc_test={_ae_cnn*100:.2f}%"
          f" | gap overfitting={(_at_cnn-_ae_cnn)*100:.2f} pp")

    # -- S2.5 Grille activation x architecture CIFAR couleur -----------
    print("\n[2.5] Grille activation x architecture CIFAR couleur (50 iter)...")
    _path_g_cf = mdl("cifar_grille_act.pt")
    if os.path.exists(_path_g_cf):
        cf_grille = charger_pt(_path_g_cf)["grille"]
    else:
        _Xtr_cpu = to_cpu(X_tr_cf2) if GPU_AVAILABLE else X_tr_cf2
        _Ytr_cpu = to_cpu(Y_tr_cf)   if GPU_AVAILABLE else Y_tr_cf
        cf_grille = {}
        for _act in ACTIVATIONS_GRILLE:
            for _anm, _arch in ARCHS_GRILLE.items():
                fixer_seeds()
                try:
                    _pg, _ = train_model(_Xtr_cpu, _Ytr_cpu, "mlp",
                                         couches_cachees=_arch, activation=_act,
                                         lr=_lr_act(0.01, _act), iters=50, batch_size=512)
                    _, _acc = evaluer_modele(X_te_cf2, y_te_cf, _pg, "mlp", activation=_act)
                    cf_grille[(_act, _anm)] = _acc
                    print(f"    {_act:10s} x {_anm:15s} -> {_acc*100:.2f}%")
                except Exception as _e:
                    print(f"    [ERREUR] {_act} x {_anm} : {_e}")
                    cf_grille[(_act, _anm)] = 0.
        sauver_pt(_path_g_cf, grille=cf_grille)

    # -- S2.6 Figures CIFAR-10 -----------------------------------------
    print("\n[2.6] Figures CIFAR-10...")

    try:
        _ordre_prog = [
            ("Lin.Gris",            "Lineaire Gris"),
            ("MLP128sig.Gris",      "MLP [128] sig. Gris"),
            ("MLP256sig.Gris",      "MLP [256] sig. Gris"),
            ("Lin.Coul",            "Lineaire Couleur"),
            ("MLP128sig.Coul",      "MLP [128] sig. Coul."),
            ("MLP256relu.Coul",     "MLP [256] relu Coul."),
            ("MLP256-128relu.Coul", "MLP [256,128] relu"),
            ("CNN4.Coul",           "CNN 4 blocs"),
        ]
        _dp = [(lbl, cf_res.get(k, {}).get("acc_test", 0) * 100) for k, lbl in _ordre_prog]
        _dp.sort(key=lambda x: x[1])
        fl, ax = plt.subplots(figsize=(14, 6))
        _cols_p = [COULEURS["vert"] if "CNN" in l else COULEURS["bleu"] for l, _ in _dp]
        ax.bar(range(len(_dp)), [v for _, v in _dp], color=_cols_p, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(_dp)))
        ax.set_xticklabels([l for l, _ in _dp], rotation=25, ha="right", fontsize=9)
        ax.axhline(78.9, color=COULEURS["rouge"],  linestyle="--", lw=1.5, label="DBN (2010) 78.9%")
        ax.axhline(90.6, color=COULEURS["violet"], linestyle="--", lw=1.5, label="Maxout (2013) 90.6%")
        _acc_mlp_max = max(
            (cf_res.get(k, {}).get("acc_test", 0) * 100 for k, _ in _ordre_prog if "CNN" not in k),
            default=0)
        _acc_cnn_v = cf_res.get("CNN4.Coul", {}).get("acc_test", 0) * 100
        if _acc_cnn_v > 0:
            ax.annotate(f"+{_acc_cnn_v - _acc_mlp_max:.1f} pp vs meilleur MLP",
                        xy=(len(_dp) - 1, _acc_cnn_v),
                        xytext=(len(_dp) - 3, _acc_cnn_v + 2),
                        arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
        ax.set_ylabel("Accuracy test (%)"); ax.set_title("Progression des modeles CIFAR-10")
        ax.legend(fontsize=9); ax.set_ylim(20, 100)
        plt.tight_layout(); plt.savefig(fig("fig_cifar_progression.pdf")); plt.close()
        print("  fig_cifar_progression.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_cifar_progression : {_e}")

    try:
        if hist_tr_cnn_cf and hist_val_cnn_cf:
            fl, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(hist_tr_cnn_cf)+1), hist_tr_cnn_cf,
                    label="Train", color=COULEURS["bleu"], linewidth=2)
            ax.plot(range(1, len(hist_val_cnn_cf)+1), hist_val_cnn_cf,
                    label="Validation", color=COULEURS["orange"], linewidth=2)
            _gap = hist_tr_cnn_cf[-1] - hist_val_cnn_cf[-1]
            ax.annotate(f"Gap={_gap:.3f}",
                        xy=(len(hist_tr_cnn_cf), hist_tr_cnn_cf[-1]),
                        xytext=(len(hist_tr_cnn_cf) * 0.6, hist_tr_cnn_cf[-1] + 0.1), fontsize=9)
            ax.set_xlabel("Epoque"); ax.set_ylabel("Perte")
            ax.set_title("Convergence CNN CIFAR-10 -- train vs validation"); ax.legend()
            plt.tight_layout(); plt.savefig(fig("fig_cifar_convergence_cnn.pdf")); plt.close()
            print("  fig_cifar_convergence_cnn.pdf OK")
        else:
            print("  fig_cifar_convergence_cnn.pdf -- ignoree (modele charge, historique absent)")
    except Exception as _e:
        print(f"  [ERREUR] fig_cifar_convergence_cnn : {_e}")

    try:
        _rng2 = np.random.RandomState(SEED)
        _idx2 = np.concatenate(
            [_rng2.choice(np.where(y_te_cf == c)[0], 100, replace=False) for c in range(10)])
        _X2d_cf = PCA(n_components=2, random_state=SEED).fit_transform(
            X_te_gray[_idx2].reshape(len(_idx2), -1))
        _y2 = y_te_cf[_idx2]
        fl, ax = plt.subplots(figsize=(10, 8))
        for _c, _col in zip(range(10), PALETTE):
            _m = _y2 == _c
            ax.scatter(_X2d_cf[_m, 0], _X2d_cf[_m, 1], c=_col, label=CIFAR_CLASSES[_c], s=15, alpha=0.6)
        ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
        ax.set_title("ACP 2D -- CIFAR-10 niveaux de gris (1000 exemples de test)")
        ax.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout(); plt.savefig(fig("fig_cifar_acp.pdf")); plt.close()
        print("  fig_cifar_acp.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_cifar_acp : {_e}")

    try:
        _hmap_cf = np.array([
            [cf_grille.get((_a, _ar), 0.) * 100 for _ar in ARCHS_GRILLE]
            for _a in ACTIVATIONS_GRILLE
        ])
        tracer_heatmap_custom(
            _hmap_cf, ACTIVATIONS_GRILLE, list(ARCHS_GRILLE.keys()),
            "Heatmap Activation x Architecture -- CIFAR-10 couleur",
            fig("fig_cifar_heatmap_activations.pdf"))
        print("  fig_cifar_heatmap_activations.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_cifar_heatmap_activations : {_e}")

    # -- S2.7 Trace CIFAR-10 -------------------------------------------
    print("\n[2.7] Trace recapitulative CIFAR-10")
    _S = "=" * 64; _L = "-" * 64
    print(f"\n{_S}"); print("CIFAR-10 -- Resultats"); print(_S)
    print(f"{'Modele':<26} {'Donnees':<9} {'Acc.Test':>9}  {'Err.Test':>8}")
    print(_L)
    for _slug, _nom, _don in [
        ("Lin.Gris",            "Lineaire",           "Gris"),
        ("MLP128relu.Gris",     "MLP [128] relu",     "Gris"),
        ("MLP256relu.Gris",     "MLP [256] relu",     "Gris"),
        ("Lin.Coul",            "Lineaire",           "Couleur"),
        ("MLP128relu.Coul",     "MLP [128] relu",     "Couleur"),
        ("MLP256relu.Coul",     "MLP [256] relu",     "Couleur"),
        ("MLP256-128relu.Coul", "MLP [256,128] relu", "Couleur"),
        ("CNN4.Coul",           "CNN 4 blocs",        "Couleur"),
    ]:
        _r = cf_res.get(_slug, {})
        print(f"{_nom:<26} {_don:<9} {_r.get('acc_test',0)*100:>8.2f}%  {_r.get('err_test',0)*100:>7.2f}%")
    print(_S)
    print("References litterature :")
    print("  Conv. Deep Belief Networks (2010) : 78.9%")
    print("  Maxout Networks (2013)            : 90.6%")
    print("  ViT (2021)                        : 99.5%")
    _best_mlp = max(
        (cf_res.get(k, {}).get("acc_test", 0)
         for k in ["MLP128sig.Coul", "MLP256relu.Coul", "MLP256-128relu.Coul"]),
        default=0)
    print(f"Gain CNN vs meilleur MLP : +{(cf_res.get('CNN4.Coul',{}).get('acc_test',0)-_best_mlp)*100:.1f} pp")

except Exception as _e_p2:
    print(f"\n[ERREUR] Partie 2 CIFAR-10 : {_e_p2}"); traceback.print_exc()

# ================================================================
# === PARTIE 3 : CBIS-DDSM ===
# ================================================================

print("\n" + "=" * 64)
print("PARTIE 3 -- CBIS-DDSM")
print("=" * 64)

try:
    # -- S3.1 Chargement 224x224 ----------------------------------------
    print("\n[3.1] Chargement CBIS-DDSM 224x224...")
    TAILLE_MAIN  = 224
    BATCH_REEL   = 8
    ACCUMULATION = 4

    X_tr_ds, y_tr_ds, X_te_ds, y_te_ds = charger_cbis_ddsm(taille=TAILLE_MAIN, utiliser_crop=True)
    _n_malins_te = int((y_te_ds == 1).sum())
    _n_benins_te = int((y_te_ds == 0).sum())
    print(f"  Train {len(X_tr_ds)} | Test {len(X_te_ds)} ({_n_malins_te} malins / {_n_benins_te} benins)")
    ds_res    = {}
    ds_scores = {}

    # -- S3.2 MLP profond -----------------------------------------------
    print("\n[3.2] MLP profond [2048,1024,512,256] ReLU...")
    _path_mlp_ds = mdl("ddsm_mlp.pt")
    X_tr_ds_flat = to_gpu(X_tr_ds.reshape(len(X_tr_ds), -1).T.astype(np.float32))
    X_te_ds_flat = X_te_ds.reshape(len(X_te_ds), -1).T.astype(np.float32)
    Y_tr_ds      = to_gpu(vector_label(y_tr_ds, n_classes=2))
    fixer_seeds()
    if os.path.exists(_path_mlp_ds):
        print(f"  Chargement depuis {_path_mlp_ds}")
        _d = charger_pt(_path_mlp_ds); params_mlp_ds, hist_mlp_ds = _d["params"], _d["hist"]
    else:
        params_mlp_ds, hist_mlp_ds = train_model(
            X_tr_ds_flat, Y_tr_ds, "mlp",
            couches_cachees=[2048, 1024, 512, 256], activation="relu",
            lr=0.003, iters=50, batch_size=64)
        sauver_pt(_path_mlp_ds, params=params_mlp_ds, hist=hist_mlp_ds)
    fixer_seeds()
    _pred_mlp, _acc_mlp = evaluer_modele(X_te_ds_flat, y_te_ds, params_mlp_ds, "mlp", activation="relu")
    _cm_mlp = matrice_confusion(y_te_ds, _pred_mlp)
    _sc_mlp = obtenir_scores(X_te_ds_flat, params_mlp_ds, "mlp", activation="relu")
    ds_scores["MLP profond"] = _sc_mlp[1, :]
    _a, _s, _sp, _fn, _fp = metriques_cm(_cm_mlp)
    ds_res["MLP profond"] = {"acc": _a, "sens": _s, "spec": _sp, "FN": _fn, "FP": _fp, "cm": _cm_mlp}
    print(f"  MLP profond | acc={_a*100:.1f}% | sens={_s*100:.1f}% | spec={_sp*100:.1f}%")
    del X_tr_ds_flat, Y_tr_ds, params_mlp_ds; liberer_gpu()

    # -- S3.3 CNN from scratch ------------------------------------------
    print("\n[3.3] CNN binaire from scratch DDSM...")
    _path_cnn_ds = mdl(f"cnn_ddsm_{TAILLE_MAIN}.pth")
    fixer_seeds()
    if os.path.exists(_path_cnn_ds):
        print(f"  Chargement depuis {_path_cnn_ds}")
        params_cnn_ds = mc.init_cnn_binaire(TAILLE_MAIN)
        params_cnn_ds["model"].load_state_dict(
            torch.load(_path_cnn_ds, map_location=params_cnn_ds["device"], weights_only=True))
        params_cnn_ds["model"].train(False)
        hist_tr_cnn_ds, hist_val_cnn_ds = [], []
    else:
        params_cnn_ds, hist_tr_cnn_ds, hist_val_cnn_ds = train_model_cnn_binaire(
            X_tr_ds, y_tr_ds, taille=TAILLE_MAIN, lr=1e-4, iters=50,
            batch_size=BATCH_REEL, accumulate_steps=ACCUMULATION, architecture="cnn_binaire")
        torch.save(params_cnn_ds["model"].state_dict(), _path_cnn_ds)
    fixer_seeds()
    _pred_cnn, _acc_cnn = evaluer_modele(X_te_ds, y_te_ds, params_cnn_ds, "cnn_binaire")
    _cm_cnn = matrice_confusion(y_te_ds, _pred_cnn)
    _sc_cnn = obtenir_scores(X_te_ds, params_cnn_ds, "cnn_binaire")
    ds_scores["CNN scratch"] = _sc_cnn[1, :]
    _a, _s, _sp, _fn, _fp = metriques_cm(_cm_cnn)
    ds_res["CNN scratch"] = {"acc": _a, "sens": _s, "spec": _sp, "FN": _fn, "FP": _fp, "cm": _cm_cnn,
                              "hist_train": hist_tr_cnn_ds, "hist_val": hist_val_cnn_ds}
    print(f"  CNN scratch | acc={_a*100:.1f}% | sens={_s*100:.1f}% | spec={_sp*100:.1f}%")
    del params_cnn_ds; liberer_gpu()

    # -- S3.4 ResNet18 --------------------------------------------------
    print("\n[3.4] ResNet18 transfer learning DDSM...")
    _path_rn_ds = mdl(f"resnet18_ddsm_{TAILLE_MAIN}.pth")
    fixer_seeds()
    if os.path.exists(_path_rn_ds):
        print(f"  Chargement depuis {_path_rn_ds}")
        params_rn_ds = mc.init_cnn_transfert(TAILLE_MAIN)
        params_rn_ds["model"].load_state_dict(
            torch.load(_path_rn_ds, map_location=params_rn_ds["device"], weights_only=True))
        params_rn_ds["model"].train(False)
        hist_tr_rn_ds, hist_val_rn_ds = [], []
    else:
        params_rn_ds, hist_tr_rn_ds, hist_val_rn_ds = train_model_cnn_binaire(
            X_tr_ds, y_tr_ds, taille=TAILLE_MAIN, lr=5e-5, iters=50,
            batch_size=BATCH_REEL, accumulate_steps=ACCUMULATION, architecture="resnet18")
        torch.save(params_rn_ds["model"].state_dict(), _path_rn_ds)
    fixer_seeds()
    _pred_rn, _acc_rn = evaluer_modele(X_te_ds, y_te_ds, params_rn_ds, "cnn_binaire")
    _cm_rn = matrice_confusion(y_te_ds, _pred_rn)
    _sc_rn = obtenir_scores(X_te_ds, params_rn_ds, "cnn_binaire")
    ds_scores["ResNet18"] = _sc_rn[1, :]
    _a, _s, _sp, _fn, _fp = metriques_cm(_cm_rn)
    ds_res["ResNet18"] = {"acc": _a, "sens": _s, "spec": _sp, "FN": _fn, "FP": _fp, "cm": _cm_rn,
                           "hist_train": hist_tr_rn_ds, "hist_val": hist_val_rn_ds}
    print(f"  ResNet18 | acc={_a*100:.1f}% | sens={_s*100:.1f}% | spec={_sp*100:.1f}%")
    del params_rn_ds; liberer_gpu()

    # -- S3.5 Calibration des seuils ------------------------------------
    print("\n[3.5] Calibration des seuils (sensibilite cible >= 80%)...")
    for _nm in ["MLP profond", "CNN scratch", "ResNet18"]:
        _sc_malin = ds_scores.get(_nm)
        if _sc_malin is None:
            continue
        fixer_seeds()
        _seuil = calibrer_seuil(y_te_ds, _sc_malin, 0.80)
        _pred_c = (_sc_malin >= _seuil).astype(int)
        _cm_c   = matrice_confusion(y_te_ds, _pred_c)
        _a_c, _s_c, _sp_c, _fn_c, _fp_c = metriques_cm(_cm_c)
        ds_res[_nm].update({"seuil_cal": _seuil, "acc_cal": _a_c,
                             "sens_cal": _s_c, "spec_cal": _sp_c,
                             "FN_cal": _fn_c, "FP_cal": _fp_c, "cm_cal": _cm_c})
        print(f"  {_nm} | seuil={_seuil:.2f} | sens_cal={_s_c*100:.1f}% | spec_cal={_sp_c*100:.1f}%")

    # -- S3.6 et S3.7 : Ablation batch=8 et batch=32 ------------------
    print("\n[3.6-3.7] Ablation DDSM 128x128 -- 2 conditions...")
    TAILLE_ABL = 128; ABL_ITERS = 25
    print(f"  Chargement donnees {TAILLE_ABL}x{TAILLE_ABL}...")
    X_tr_abl, y_tr_abl, X_te_abl, y_te_abl = charger_cbis_ddsm(taille=TAILLE_ABL, utiliser_crop=True)

    _configs_abl = [
        {"nom": "1. Baseline",
         "use_bn": False, "use_dropout": False, "weight_decay": 0.0,  "n_classes": 2},
        {"nom": "2. +BatchNorm",
         "use_bn": True,  "use_dropout": False, "weight_decay": 0.0,  "n_classes": 2, "bn_momentum": 0.1},
        {"nom": "3. +Dropout seul",
         "use_bn": False, "use_dropout": True,  "weight_decay": 0.0,  "n_classes": 2},
        {"nom": "4. +BN+Drop+WD",
         "use_bn": True,  "use_dropout": True,  "weight_decay": 1e-4, "n_classes": 2, "bn_momentum": 0.1},
        {"nom": "5. +BCE",
         "use_bn": True,  "use_dropout": True,  "weight_decay": 1e-4, "n_classes": 1, "bn_momentum": 0.1},
        {"nom": "6. BN mom=0.01",
         "use_bn": True,  "use_dropout": True,  "weight_decay": 1e-4, "n_classes": 2, "bn_momentum": 0.01},
        {"nom": "7. MixUp a=0.4",
         "use_bn": True,  "use_dropout": True,  "weight_decay": 1e-4, "n_classes": 2, "bn_momentum": 0.01,
         "use_mixup": True, "mixup_alpha": 0.4},
    ]

    def _run_ablation(batch_sz, accum, suffix):
        _res, _htr, _hval = [], [], []
        for _cfg in _configs_abl:
            _idx_cfg = _cfg["nom"].split(".")[0].strip()
            _path_a  = mdl(f"ddsm_abl_{_idx_cfg}_b{suffix}.pt")
            fixer_seeds()
            if os.path.exists(_path_a):
                _da = charger_pt(_path_a)
                _ht, _hv, _r = _da["ht"], _da["hv"], _da["r"]
            else:
                try:
                    _pa, _ht, _hv = train_ablation_cnn(
                        X_tr_abl, y_tr_abl,
                        taille=TAILLE_ABL, config=_cfg,
                        lr=1e-4, iters=ABL_ITERS,
                        batch_size=batch_sz, accumulate_steps=accum)
                    _pred_a, _acc_a = evaluer_modele(X_te_abl, y_te_abl, _pa, "cnn_configurable")
                    _cm_a = matrice_confusion(y_te_abl, _pred_a)
                    _, _sa, _spa, _, _ = metriques_cm(_cm_a)
                    _r = {"acc": _acc_a, "sens": _sa, "spec": _spa,
                          "L0": (_ht[0] if _ht else float("nan")),
                          "comportement": detecter_trivial(_sa, _spa),
                          "nom": _cfg["nom"]}
                    sauver_pt(_path_a, ht=_ht, hv=_hv, r=_r)
                    del _pa; liberer_gpu()
                except Exception as _ea:
                    print(f"    [ERREUR] {_cfg['nom']} b{suffix}: {_ea}")
                    _ht, _hv = [], []
                    _r = {"acc": 0., "sens": 0., "spec": 0., "L0": float("nan"),
                          "comportement": "Erreur", "nom": _cfg["nom"]}
            _res.append(_r); _htr.append(_ht); _hval.append(_hv)
            print(f"    {_cfg['nom']:<30} acc={_r['acc']*100:.1f}% "
                  f"sens={_r['sens']*100:.1f}% L0={_r['L0']:.2f} -> {_r['comportement']}")
        return _res, _htr, _hval

    print("\n  -- Condition A : batch physique=8, accumulation=4 --")
    abl_res_8,  abl_htr_8,  abl_hval_8  = _run_ablation(8,  4, "8")
    print("\n  -- Condition B : batch physique=32, accumulation=1 --")
    abl_res_32, abl_htr_32, abl_hval_32 = _run_ablation(32, 1, "32")

    # -- S3.8 Figures DDSM ---------------------------------------------
    print("\n[3.8] Figures DDSM...")

    try:
        _rn = ds_res.get("ResNet18", {})
        if "cm" in _rn and "cm_cal" in _rn:
            tracer_double_confusion(_rn["cm"], _rn["cm_cal"], _rn.get("seuil_cal", 0.5),
                                    "ResNet18 -- avant/apres calibration du seuil",
                                    fig("fig_ddsm_confusion_resnet18.pdf"))
            print("  fig_ddsm_confusion_resnet18.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_ddsm_confusion_resnet18 : {_e}")

    try:
        _ml = ds_res.get("MLP profond", {})
        if "cm" in _ml and "cm_cal" in _ml:
            tracer_double_confusion(_ml["cm"], _ml["cm_cal"], _ml.get("seuil_cal", 0.5),
                                    "MLP profond -- avant/apres calibration du seuil",
                                    fig("fig_ddsm_confusion_mlp.pdf"))
            print("  fig_ddsm_confusion_mlp.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_ddsm_confusion_mlp : {_e}")

    try:
        fl, ax = plt.subplots(figsize=(8, 7))
        for _nm_roc, _col_roc in zip(
            ["MLP profond", "CNN scratch", "ResNet18"],
            [COULEURS["bleu"], COULEURS["orange"], COULEURS["vert"]]
        ):
            _sc = ds_scores.get(_nm_roc)
            if _sc is not None:
                _fpr, _tpr, _ = roc_curve(y_te_ds, _sc)
                _auc_v = sklearn_auc(_fpr, _tpr)
                ax.plot(_fpr, _tpr, color=_col_roc, linewidth=2,
                        label=f"{_nm_roc} (AUC={_auc_v:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatoire")
        ax.set_xlabel("Taux faux positifs"); ax.set_ylabel("Taux vrais positifs")
        ax.set_title("Courbes ROC -- CBIS-DDSM"); ax.legend(loc="lower right")
        plt.tight_layout(); plt.savefig(fig("fig_ddsm_roc.pdf")); plt.close()
        print("  fig_ddsm_roc.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_ddsm_roc : {_e}")

    def _tracer_abl_conv(htr, hval, labels, batch_sz, save_path):
        fl, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
        for _ht, _hv, _lbl, _col in zip(htr, hval, labels, PALETTE):
            if _ht:
                ax_t.plot(range(1, len(_ht)+1), _ht, label=_lbl, color=_col, linewidth=1.5)
            if _hv:
                ax_v.plot(range(1, len(_hv)+1), _hv, label=_lbl, color=_col, linewidth=1.5)
        ax_t.set_title(f"Train -- batch={batch_sz}")
        ax_t.set_xlabel("Epoque"); ax_t.set_ylabel("Perte")
        ax_v.set_title(f"Validation -- batch={batch_sz}"); ax_v.set_xlabel("Epoque")
        ax_t.legend(fontsize=7, loc="upper right"); ax_v.legend(fontsize=7, loc="upper right")
        fl.suptitle(f"Ablation CBIS-DDSM 128x128 -- batch physique={batch_sz}", fontsize=13)
        plt.tight_layout(); plt.savefig(save_path); plt.close()

    try:
        _lbl8 = [_r["nom"] for _r in abl_res_8]
        _tracer_abl_conv(abl_htr_8, abl_hval_8, _lbl8, 8,
                         fig("fig_ddsm_ablation_convergence_batch8.pdf"))
        print("  fig_ddsm_ablation_convergence_batch8.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_ddsm_ablation_convergence_batch8 : {_e}")

    try:
        _lbl32 = [_r["nom"] for _r in abl_res_32]
        _tracer_abl_conv(abl_htr_32, abl_hval_32, _lbl32, 32,
                         fig("fig_ddsm_ablation_convergence_batch32.pdf"))
        print("  fig_ddsm_ablation_convergence_batch32.pdf OK")
    except Exception as _e:
        print(f"  [ERREUR] fig_ddsm_ablation_convergence_batch32 : {_e}")

    # -- S3.9 Trace DDSM -----------------------------------------------
    print("\n[3.9] Trace recapitulative DDSM")
    _S = "=" * 64; _L = "-" * 64

    print(f"\n{_S}"); print("CBIS-DDSM -- Resultats principaux (seuil=0.50)"); print(_S)
    print(f"{'Modele':<16} {'Acc.':>6}  {'Sensib.':>7}  {'Specif.':>7}  {'FN':>9}  {'FP':>9}")
    print(_L)
    for _nm in ["MLP profond", "CNN scratch", "ResNet18"]:
        _r = ds_res.get(_nm, {})
        print(f"{_nm:<16} {_r.get('acc',0)*100:>5.1f}%  "
              f"{_r.get('sens',0)*100:>6.1f}%  {_r.get('spec',0)*100:>6.1f}%  "
              f"{str(_r.get('FN','?'))+'/' + str(_n_malins_te):>9}  "
              f"{str(_r.get('FP','?'))+'/' + str(_n_benins_te):>9}")
    print(_S)

    print(f"\n{_S}"); print("CBIS-DDSM -- Apres calibration (sensibilite cible >= 80%)"); print(_S)
    print(f"{'Modele':<16} {'Seuil':>6}  {'Sensib.':>7}  {'Specif.':>7}  {'FN':>9}  {'FP':>9}")
    print(_L)
    for _nm in ["MLP profond", "CNN scratch", "ResNet18"]:
        _r = ds_res.get(_nm, {})
        print(f"{_nm:<16} {_r.get('seuil_cal',0):>6.2f}  "
              f"{_r.get('sens_cal',0)*100:>6.1f}%  {_r.get('spec_cal',0)*100:>6.1f}%  "
              f"{str(_r.get('FN_cal','?'))+'/' + str(_n_malins_te):>9}  "
              f"{str(_r.get('FP_cal','?'))+'/' + str(_n_benins_te):>9}")
    print(_S)
    print("Reference litterature : ResNet-50 ~85-88% (dataset >= 5000 images)")

    print(f"\n{_S}"); print("ABLATION -- Resume comparatif batch=8 vs batch=32"); print(_S)
    print(f"{'Config':<22}  {'--- Batch=8 ---':^20}   {'--- Batch=32 ---':^20}")
    print(f"{'':22}  {'Acc.':>5} {'Sensib.':>7} {'L0':>5}   {'Acc.':>5} {'Sensib.':>7} {'L0':>5}")
    print(_L)
    for _r8, _r32 in zip(abl_res_8, abl_res_32):
        print(f"{_r8['nom']:<22}  {_r8['acc']*100:>4.1f}% {_r8['sens']*100:>6.1f}% {_r8['L0']:>5.2f}   "
              f"{_r32['acc']*100:>4.1f}% {_r32['sens']*100:>6.1f}% {_r32['L0']:>5.2f}")
    print(_S)
    print("\nComportements triviaux detectes :")
    for _i, (_r8, _r32) in enumerate(zip(abl_res_8, abl_res_32), 1):
        if _r8["comportement"] != "Equilibre":
            print(f"  Batch=8  : Config {_i} -> {_r8['comportement']}")
        if _r32["comportement"] != "Equilibre":
            print(f"  Batch=32 : Config {_i} -> {_r32['comportement']}")
    _l0_b32_c2 = abl_res_32[1]["L0"] if len(abl_res_32) > 1 else float("nan")
    _l0_b8_c2  = abl_res_8[1]["L0"]  if len(abl_res_8) > 1  else float("nan")
    _hyp = "validee" if _l0_b32_c2 < _l0_b8_c2 else "infirmee"
    print(f"Conclusion BN : hypothese {_hyp} -- perte initiale batch=32 config2 : {_l0_b32_c2:.2f}"
          f" vs batch=8 : {_l0_b8_c2:.2f}")

except Exception as _e_p3:
    print(f"\n[ERREUR] Partie 3 DDSM : {_e_p3}"); traceback.print_exc()

# ================================================================
# === SYNTHESE FINALE ===
# ================================================================

print("\n" + "=" * 64)
print("SYNTHESE FINALE -- Figures generees")
print("=" * 64)

_figures = [
    "fig_mnist_convergence.pdf",
    "fig_mnist_tailles_couches.pdf",
    "fig_mnist_acp.pdf",
    "fig_mnist_erreurs_communes.pdf",
    "fig_mnist_heatmap_activations.pdf",
    "fig_cifar_filtres.pdf",
    "fig_cifar_progression.pdf",
    "fig_cifar_convergence_cnn.pdf",
    "fig_cifar_acp.pdf",
    "fig_cifar_heatmap_activations.pdf",
    "fig_ddsm_confusion_resnet18.pdf",
    "fig_ddsm_confusion_mlp.pdf",
    "fig_ddsm_roc.pdf",
    "fig_ddsm_ablation_convergence_batch8.pdf",
    "fig_ddsm_ablation_convergence_batch32.pdf",
]
_ok = 0
for _f in _figures:
    _p = fig(_f)
    if os.path.exists(_p):
        print(f"  OK      {_f}"); _ok += 1
    else:
        print(f"  MANQUANT  {_f}")
print(f"\n  {_ok}/{len(_figures)} figures OK")
print(f"  Figures : {FIGURES_DIR}")
print(f"  Trace   : {LOG_PATH}")

_log.close()
