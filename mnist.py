import numpy as np
from sklearn.decomposition import PCA

import utils.sorties as sorties
from utils.donnees import charger_mnist
from utils.preprocessing import vector_label
from utils.entrainement import train_model
from utils.evaluation import evaluer_modele
from utils.visualisation import (
    visualiser_exemples,
    visualiser_erreurs,
    visualiser_erreurs_communes,
    tracer_convergence,
    tracer_acp,
)

S = lambda nom: sorties.chemin("mnist", nom)


def nb_params(model_type, n_h1=64, n_h2=None):
    if model_type == "linear":
        return 784 * 10 + 10
    elif n_h2 is None:
        return 784 * n_h1 + n_h1 + n_h1 * 10 + 10
    else:
        return 784 * n_h1 + n_h1 + n_h1 * n_h2 + n_h2 + n_h2 * 10 + 10


log = sorties.demarrer_log("mnist")
try:
    # ── Données ───────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = charger_mnist()
    visualiser_exemples(X_train, y_train, titre='Exemples MNIST', image_shape=(28, 28))

    Y_train = vector_label(y_train)

    # ── Entraînement ──────────────────────────────────────────────
    params_lin, hist_lin = train_model(X_train, Y_train, "linear", lr=0.1, iters=100, batch_size=256)
    params_h1,  hist_h1  = train_model(X_train, Y_train, "mlp", n_h1=64, lr=0.1, iters=100, batch_size=256)
    params_h2,  hist_h2  = train_model(X_train, Y_train, "mlp", n_h1=64, n_h2=32, lr=0.1, iters=100, batch_size=256)

    # ── Évaluation TRAIN ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUR LE TRAIN")
    print("=" * 60)
    _, acc_lin_train = evaluer_modele(X_train, y_train, params_lin, "linear", ensemble="TRAIN")
    _, acc_h1_train  = evaluer_modele(X_train, y_train, params_h1,  "mlp",    ensemble="TRAIN")
    _, acc_h2_train  = evaluer_modele(X_train, y_train, params_h2,  "mlp", n_h2=True, ensemble="TRAIN")

    # ── Évaluation TEST ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ET ANALYSE DES ERREURS DE CLASSIFICATION")
    print("=" * 60)
    pred_lin, acc_lin = evaluer_modele(X_test, y_test, params_lin, "linear", afficher_exemples=True)
    pred_h1,  acc_h1  = evaluer_modele(X_test, y_test, params_h1,  "mlp",   afficher_exemples=True)
    pred_h2,  acc_h2  = evaluer_modele(X_test, y_test, params_h2,  "mlp", n_h2=True, afficher_exemples=True)

    # ── Tableau récapitulatif ─────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Modèle':<20} {'Nb params':>10} {'Err. Train':>12} {'Err. Test':>12}")
    print("-" * 70)
    print(f"{'Linéaire':<20} {nb_params('linear'):>10} {(1-acc_lin_train)*100:>11.2f}% {(1-acc_lin)*100:>11.2f}%")
    print(f"{'MLP H=1':<20} {nb_params('mlp',64):>10} {(1-acc_h1_train)*100:>11.2f}% {(1-acc_h1)*100:>11.2f}%")
    print(f"{'MLP H=2':<20} {nb_params('mlp',64,32):>10} {(1-acc_h2_train)*100:>11.2f}% {(1-acc_h2)*100:>11.2f}%")
    print("=" * 70)

    # ── Visualisations ────────────────────────────────────────────
    visualiser_erreurs(X_test, y_test, pred_lin,
                       titre="Modèle Linéaire", save_path=S("erreurs_lineaire.png"))
    visualiser_erreurs(X_test, y_test, pred_h1,
                       titre="MLP H=1", save_path=S("erreurs_mlp_h1.png"))
    visualiser_erreurs(X_test, y_test, pred_h2,
                       titre="MLP H=2", save_path=S("erreurs_mlp_h2.png"))

    visualiser_erreurs_communes(X_test, y_test, {
        "Linéaire": pred_lin,
        "MLP H=1":  pred_h1,
        "MLP H=2":  pred_h2,
    }, save_path=S("erreurs_communes.png"))

    acp = PCA(n_components=2)
    acp.fit(X_train.T)
    X_test_2d = acp.transform(X_test.T)

    tracer_convergence(
        [hist_lin, hist_h1, hist_h2],
        ["Linéaire", "MLP H=1", "MLP H=2"],
        titre="Convergence MNIST (Log Loss)",
        save_path=S("convergence.png"),
    )
    tracer_acp(
        X_test_2d, y_test,
        titre="Visualisation ACP des chiffres MNIST",
        save_path=S("acp.png"),
    )

finally:
    log.close()
