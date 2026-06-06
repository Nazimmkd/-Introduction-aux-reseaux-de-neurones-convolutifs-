"""
ddsm.py — Classification binaire bénin vs malin sur CBIS-DDSM
Architectures : MLP ReLU | CNN binaire | ResNet18 transfer learning
Sorties       : sorties/ddsm/
"""
import numpy as np
import os

import utils.sorties as sorties
from utils.donnees_ddsm import charger_cbis_ddsm
from utils.preprocessing import vector_label
from utils.entrainement import train_model, train_model_cnn_binaire, train_ablation_cnn
from utils.evaluation import evaluer_modele, matrice_confusion, obtenir_scores, calibrer_seuil
from utils.gpu import to_gpu, liberer_gpu
from utils.visualisation import tracer_convergence, tracer_confusion

S = lambda nom: sorties.chemin("ddsm", nom)
M = lambda nom: sorties.chemin("modeles", nom)

TAILLE            = 224    # résolution native ResNet18 — meilleure conservation de texture
CIBLE_SENSIBILITE = 0.80   # garantir 80% de détection des cancers
BATCH_REEL        = 8      # taille réelle du batch GPU (limité VRAM à 224×224)
ACCUMULATION      = 4      # batch effectif = 8 × 4 = 32 (sans saturation mémoire)
CNN_WEIGHTS       = M(f"cnn_ddsm_{TAILLE}.pth")
RESNET_WEIGHTS    = M(f"resnet18_ddsm_{TAILLE}.pth")


log = sorties.demarrer_log("ddsm")
try:

    # ── Chargement (crops centrés sur la masse) ───────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION BINAIRE CBIS-DDSM — benin vs malin")
    print("=" * 60)

    X_train, y_train, X_test, y_test = charger_cbis_ddsm(taille=TAILLE, utiliser_crop=True)

    # ── MLP ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MLP — CBIS-DDSM (crop)")
    print("=" * 60)

    X_tr_flat = to_gpu(X_train.reshape(len(X_train), -1).T.astype(np.float32))
    X_te_flat = X_test.reshape(len(X_test), -1).T.astype(np.float32)
    Y_tr      = to_gpu(vector_label(y_train, n_classes=2))

    params_mlp, hist_mlp = train_model(
        X_tr_flat, Y_tr, "mlp",
        couches_cachees=[2048, 1024, 512, 256],
        activation="relu",
        lr=0.003, iters=50, batch_size=64,
    )
    pred_mlp, acc_mlp = evaluer_modele(X_te_flat, y_test, params_mlp, "mlp", activation="relu")
    cm_mlp = matrice_confusion(y_test, pred_mlp)

    scores_mlp   = obtenir_scores(X_te_flat, params_mlp, "mlp", activation="relu")
    seuil_mlp    = calibrer_seuil(y_test, scores_mlp[1, :], CIBLE_SENSIBILITE)
    pred_mlp_cal = (scores_mlp[1, :] >= seuil_mlp).astype(int)
    print(f"\n  Matrice apres calibration du seuil :")
    cm_mlp_cal   = matrice_confusion(y_test, pred_mlp_cal)

    del X_tr_flat, Y_tr, params_mlp
    liberer_gpu()

    # ── CNN binaire (entraîné de zéro) ────────────────────────────
    print("\n" + "=" * 60)
    print("CNN binaire (from scratch) — CBIS-DDSM (crop)")
    print("=" * 60)

    import torch as _torch
    if os.path.exists(CNN_WEIGHTS):
        print(f"[LOAD] CNN existant — {CNN_WEIGHTS}")
        import modeles.modele_convolutif as mc
        params_cnn = mc.init_cnn_binaire(TAILLE)
        params_cnn['model'].load_state_dict(
            _torch.load(CNN_WEIGHTS, map_location=params_cnn['device'], weights_only=True)
        )
        params_cnn['model'].train(False)
        hist_train_cnn, hist_val_cnn = [], []
    else:
        params_cnn, hist_train_cnn, hist_val_cnn = train_model_cnn_binaire(
            X_train, y_train, taille=TAILLE, lr=1e-4, iters=50,
            batch_size=BATCH_REEL, accumulate_steps=ACCUMULATION,
            architecture='cnn_binaire',
        )
        _torch.save(params_cnn['model'].state_dict(), CNN_WEIGHTS)
        print(f"[SAVE] CNN sauvegarde dans {CNN_WEIGHTS}")

    pred_cnn, acc_cnn = evaluer_modele(X_test, y_test, params_cnn, "cnn_binaire")
    cm_cnn = matrice_confusion(y_test, pred_cnn)

    scores_cnn   = obtenir_scores(X_test, params_cnn, "cnn_binaire")
    seuil_cnn    = calibrer_seuil(y_test, scores_cnn[1, :], CIBLE_SENSIBILITE)
    pred_cnn_cal = (scores_cnn[1, :] >= seuil_cnn).astype(int)
    print(f"\n  Matrice apres calibration du seuil :")
    cm_cnn_cal   = matrice_confusion(y_test, pred_cnn_cal)

    # ── ResNet18 transfer learning ────────────────────────────────
    print("\n" + "=" * 60)
    print("ResNet18 transfer learning — CBIS-DDSM (crop)")
    print("=" * 60)

    if os.path.exists(RESNET_WEIGHTS):
        print(f"[LOAD] ResNet18 existant — {RESNET_WEIGHTS}")
        import modeles.modele_convolutif as mc
        params_rn = mc.init_cnn_transfert(TAILLE)
        params_rn['model'].load_state_dict(
            _torch.load(RESNET_WEIGHTS, map_location=params_rn['device'], weights_only=True)
        )
        params_rn['model'].train(False)
        hist_train_rn, hist_val_rn = [], []
    else:
        params_rn, hist_train_rn, hist_val_rn = train_model_cnn_binaire(
            X_train, y_train, taille=TAILLE, lr=5e-5, iters=50,
            batch_size=BATCH_REEL, accumulate_steps=ACCUMULATION,
            architecture='resnet18',
        )
        _torch.save(params_rn['model'].state_dict(), RESNET_WEIGHTS)
        print(f"[SAVE] ResNet18 sauvegarde dans {RESNET_WEIGHTS}")

    pred_rn, acc_rn = evaluer_modele(X_test, y_test, params_rn, "cnn_binaire")
    cm_rn = matrice_confusion(y_test, pred_rn)

    scores_rn   = obtenir_scores(X_test, params_rn, "cnn_binaire")
    seuil_rn    = calibrer_seuil(y_test, scores_rn[1, :], CIBLE_SENSIBILITE)
    pred_rn_cal = (scores_rn[1, :] >= seuil_rn).astype(int)
    print(f"\n  Matrice apres calibration du seuil :")
    cm_rn_cal   = matrice_confusion(y_test, pred_rn_cal)

    # ── Résumé ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTATS FINAUX — seuil=0.50")
    print("=" * 60)
    print(f"  MLP [512,256] ReLU  : {acc_mlp * 100:.2f}%")
    print(f"  CNN from scratch    : {acc_cnn * 100:.2f}%")
    print(f"  ResNet18 transfer   : {acc_rn  * 100:.2f}%")
    print()
    print("  References litterature :")
    print("    ResNet-50 (transfer) : ~85-88%")
    print("    CNN from scratch     : ~70-75%")

    # ── Visualisations ────────────────────────────────────────────
    tracer_confusion(cm_mlp,     ["Benin", "Malin"],
                     titre="MLP — seuil 0.5",
                     save_path=S("mlp_confusion_05.png"))
    tracer_confusion(cm_mlp_cal, ["Benin", "Malin"],
                     titre=f"MLP — seuil calibre {seuil_mlp:.2f} (sensibilite {CIBLE_SENSIBILITE*100:.0f}%)",
                     save_path=S("mlp_confusion_cal.png"))

    tracer_confusion(cm_cnn,     ["Benin", "Malin"],
                     titre="CNN — seuil 0.5",
                     save_path=S("cnn_confusion_05.png"))
    tracer_confusion(cm_cnn_cal, ["Benin", "Malin"],
                     titre=f"CNN — seuil calibre {seuil_cnn:.2f} (sensibilite {CIBLE_SENSIBILITE*100:.0f}%)",
                     save_path=S("cnn_confusion_cal.png"))

    tracer_confusion(cm_rn,      ["Benin", "Malin"],
                     titre="ResNet18 — seuil 0.5",
                     save_path=S("resnet_confusion_05.png"))
    tracer_confusion(cm_rn_cal,  ["Benin", "Malin"],
                     titre=f"ResNet18 — seuil calibre {seuil_rn:.2f} (sensibilite {CIBLE_SENSIBILITE*100:.0f}%)",
                     save_path=S("resnet_confusion_cal.png"))

    tracer_convergence([hist_mlp], ["MLP [512,256]"],
                       titre="CBIS-DDSM — Convergence MLP",
                       save_path=S("mlp_convergence.png"))

    if hist_train_cnn:
        tracer_convergence([hist_train_cnn, hist_val_cnn], ["CNN train", "CNN val"],
                           titre="CBIS-DDSM — Convergence CNN",
                           save_path=S("cnn_convergence.png"))

    if hist_train_rn:
        tracer_convergence([hist_train_rn, hist_val_rn], ["ResNet18 train", "ResNet18 val"],
                           titre="CBIS-DDSM — Convergence ResNet18",
                           save_path=S("resnet_convergence.png"))

    # ── Ablation study ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION STUDY — BN / Dropout / Weight Decay / BCE vs CE")
    print("=" * 60)

    # Résolution réduite (128px) pour l'ablation : 5 entraînements successifs
    TAILLE_ABL = 128
    ABL_ITERS  = 25

    print(f"\n[LOAD] Données {TAILLE_ABL}×{TAILLE_ABL} pour ablation (cache séparé)...")
    X_abl_tr, y_abl_tr, X_abl_te, y_abl_te = charger_cbis_ddsm(
        taille=TAILLE_ABL, utiliser_crop=True)

    # Chaque config active une technique de plus (ablation séquentielle)
    configs_abl = [
        {'nom': '1. Baseline — aucune regularisation',
         'use_bn': False, 'use_dropout': False, 'weight_decay': 0.0,  'n_classes': 2},
        {'nom': '2. + BatchNorm',
         'use_bn': True,  'use_dropout': False, 'weight_decay': 0.0,  'n_classes': 2},
        {'nom': '3. + Dropout',
         'use_bn': False, 'use_dropout': True,  'weight_decay': 0.0,  'n_classes': 2},
        {'nom': '4. + BN + Dropout + Weight Decay (CE)',
         'use_bn': True,  'use_dropout': True,  'weight_decay': 1e-4, 'n_classes': 2},
        {'nom': '5. Complet avec BCE (sigmoid, 1 sortie)',
         'use_bn': True,  'use_dropout': True,  'weight_decay': 1e-4, 'n_classes': 1},
        # Perspectives d'amélioration pour la soutenance
        {'nom': '6. BN momentum=0.01 (stats stables petit batch)',
         'use_bn': True,  'use_dropout': True,  'weight_decay': 1e-4, 'n_classes': 2,
         'bn_momentum': 0.01},
        {'nom': '7. BN momentum=0.01 + MixUp alpha=0.4',
         'use_bn': True,  'use_dropout': True,  'weight_decay': 1e-4, 'n_classes': 2,
         'bn_momentum': 0.01, 'use_mixup': True, 'mixup_alpha': 0.4},
    ]

    resultats_abl  = []
    hists_train_abl, hists_val_abl, labels_abl = [], [], []

    for cfg in configs_abl:
        params_abl, h_tr, h_val = train_ablation_cnn(
            X_abl_tr, y_abl_tr,
            taille=TAILLE_ABL, config=cfg,
            lr=1e-4, iters=ABL_ITERS,
            batch_size=BATCH_REEL, accumulate_steps=ACCUMULATION,
        )
        pred_abl, acc_abl = evaluer_modele(
            X_abl_te, y_abl_te, params_abl, "cnn_configurable")
        cm_abl = matrice_confusion(y_abl_te, pred_abl)
        resultats_abl.append({'nom': cfg['nom'], 'acc': acc_abl, 'cm': cm_abl})
        hists_train_abl.append(h_tr)
        hists_val_abl.append(h_val)
        labels_abl.append(cfg['nom'])
        del params_abl
        liberer_gpu()

    print("\n" + "=" * 60)
    print("RÉSUMÉ ABLATION")
    print("=" * 60)
    for r in resultats_abl:
        print(f"  {r['nom']:<48} : {r['acc'] * 100:.2f}%")

    tracer_convergence(
        hists_train_abl, labels_abl,
        titre="Ablation Study — Perte entraînement (CBIS-DDSM 128×128)",
        save_path=S("ablation_train_convergence.png"),
    )
    tracer_convergence(
        hists_val_abl, labels_abl,
        titre="Ablation Study — Perte validation (CBIS-DDSM 128×128)",
        save_path=S("ablation_val_convergence.png"),
    )

finally:
    log.close()
