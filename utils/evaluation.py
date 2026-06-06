import numpy as np
import modeles.modele_lineaire as ml
import modeles.modele_couches_cachées as mcc
import modeles.modele_convolutif as mc
from utils.gpu import to_cpu


def evaluer_modele(X, y, params, model_type="linear", n_h2=False, ensemble="TEST",
                   afficher_exemples=False, activation="relu"):
    """
    Calcule les prédictions et la précision d'un modèle.

    X           : données d'entrée (numpy ou CuPy)
    y           : labels entiers (numpy)
    params      : paramètres du modèle (liste pour linéaire, dict pour mlp/cnn)
    model_type  : "linear" | "mlp" | "cnn"
    activation  : fonction d'activation utilisée lors de l'entraînement
                  ("sigmoid" | "tanh" | "relu" | "heaviside") — doit correspondre
                  à celle passée à train_model
    n_h2        : conservé pour rétrocompatibilité (ignoré, calculé depuis params)
    ensemble    : "TRAIN" ou "TEST" — affiché dans le rapport
    """
    if model_type == "linear":
        Z      = ml.fonction_score(X, params[0], params[1])
        scores = ml.softmax(Z)
    elif model_type == "mlp":
        L      = len(params) // 2
        cache  = mcc.forward_pass(X, params, activation)
        scores = cache[f"A{L}"]
    elif model_type == "cnn":
        scores, _ = mc.cnn_forward(X, params)
    elif model_type == "cnn_binaire":
        scores, _ = mc.cnn_binaire_forward(X, params)
    elif model_type == "cnn_configurable":
        scores, _ = mc.cnn_configurable_forward(X, params)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")

    scores      = to_cpu(scores)
    y           = to_cpu(y)
    predictions = np.argmax(scores, axis=0)
    probs       = np.max(scores, axis=0)
    error_idx   = np.where(predictions != y)[0]
    accuracy    = np.mean(predictions == y)

    if model_type == "mlp":
        n_hidden = len(params) // 2 - 1
        label = f"[MLP {n_hidden} couche(s) cachée(s) — {activation}]"
    elif model_type == "cnn":
        label = "[CNN PyTorch]"
    elif model_type == "cnn_binaire":
        label = "[CNN binaire PyTorch]"
    else:
        label = "[Linéaire]"

    print(f"\n{label} (Ensemble {ensemble})")
    print(f"Precision: {accuracy * 100:.2f}%")
    print(f"Nombre d'erreurs: {len(error_idx)} ({len(error_idx) / len(y) * 100:.2f}%)")
    print(f"Predictions correctes: {len(y) - len(error_idx)} ({(1 - len(error_idx) / len(y)) * 100:.2f}%)")

    if afficher_exemples and len(error_idx) > 0 and ensemble == "TEST":
        print("\nExemples d'erreurs :")
        for i in error_idx[:5]:
            print(f"  Indice {i}: Vraie classe={y[i]}, Prediction={predictions[i]}, Confiance={probs[i] * 100:.2f}%")

    return predictions, accuracy


def obtenir_scores(X, params, model_type, activation="relu"):
    """
    Retourne la matrice de scores bruts (n_classes, N) sans afficher les métriques.
    Utilisé pour la calibration du seuil de décision.
    """
    if model_type == "linear":
        Z = ml.fonction_score(X, params[0], params[1])
        return to_cpu(ml.softmax(Z))
    elif model_type == "mlp":
        L     = len(params) // 2
        cache = mcc.forward_pass(X, params, activation)
        return to_cpu(cache[f"A{L}"])
    elif model_type == "cnn":
        scores, _ = mc.cnn_forward(X, params)
        return to_cpu(scores)
    elif model_type in ("cnn_binaire", "cnn_transfert"):
        scores, _ = mc.cnn_binaire_forward(X, params)
        return to_cpu(scores)
    elif model_type == "cnn_configurable":
        scores, _ = mc.cnn_configurable_forward(X, params)
        return to_cpu(scores)
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}")


def calibrer_seuil(y_true, scores_malin, cible_sensibilite=0.80):
    """
    Trouve le seuil de décision qui atteint la sensibilité cible (% de malins détectés)
    tout en maximisant la précision globale.

    En médecine, on préfère rater 0 cancer (FN=0) au prix de plus de biopsies inutiles.
    Par défaut, la cible est 80% de sensibilité.
    """
    meilleur = {'seuil': 0.5, 'acc': 0.0, 'sensi': 0.0, 'speci': 0.0}
    for s in np.linspace(0.05, 0.95, 181):
        pred = (scores_malin >= s).astype(int)
        TP   = int(((pred == 1) & (y_true == 1)).sum())
        FN   = int(((pred == 0) & (y_true == 1)).sum())
        TN   = int(((pred == 0) & (y_true == 0)).sum())
        FP   = int(((pred == 1) & (y_true == 0)).sum())
        sensi = TP / max(TP + FN, 1)
        speci = TN / max(TN + FP, 1)
        acc   = (TP + TN) / max(len(y_true), 1)
        if sensi >= cible_sensibilite and acc > meilleur['acc']:
            meilleur = {'seuil': float(s), 'acc': acc, 'sensi': sensi, 'speci': speci}

    s = meilleur['seuil']
    print(f"\n  Calibration seuil (sensibilite >= {cible_sensibilite*100:.0f}%) :")
    print(f"    Seuil optimal : {s:.2f}  (defaut : 0.50)")
    print(f"    Precision     : {meilleur['acc']  * 100:.1f}%")
    print(f"    Sensibilite   : {meilleur['sensi'] * 100:.1f}%")
    print(f"    Specificite   : {meilleur['speci'] * 100:.1f}%")
    return s


def matrice_confusion(y_true, y_pred, classes=("Bénin", "Malin")):
    """
    Calcule la matrice de confusion et affiche les métriques médicales.
    Pour la classification binaire, les faux négatifs (cancer manqué) sont critiques.
    Retourne la matrice numpy (n_classes, n_classes).
    """
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(to_cpu(y_true), to_cpu(y_pred)):
        cm[int(t), int(p)] += 1

    print(f"\n  Matrice de confusion ({' / '.join(classes)})")
    print("  " + "-" * (12 * n + 14))
    entete = f"  {'':12}" + "".join(f"{'Prédit ' + c:>12}" for c in classes)
    print(entete)
    for i, classe in enumerate(classes):
        ligne = f"  {'Réel ' + classe:<12}" + "".join(f"{cm[i, j]:>12}" for j in range(n))
        print(ligne)
    print("  " + "-" * (12 * n + 14))

    # Métriques médicales (classification binaire bénin=0, malin=1)
    if n == 2:
        TP = cm[1, 1]
        FN = cm[1, 0]  # malin prédit bénin = cancer manqué
        FP = cm[0, 1]  # bénin prédit malin = biopsie inutile
        TN = cm[0, 0]

        sensibilite = TP / max(TP + FN, 1)
        specificite = TN / max(TN + FP, 1)
        precision   = TP / max(TP + FP, 1)

        print(f"\n  Sensibilité (rappel malin)        : {sensibilite * 100:.1f}%")
        print(f"  Spécificité (rappel bénin)        : {specificite * 100:.1f}%")
        print(f"  Précision (positive predictive)   : {precision   * 100:.1f}%")
        print(f"  Faux négatifs (cancers manqués)   : {FN}  ({FN / max(TP + FN, 1) * 100:.1f}%)")
        print(f"  Faux positifs (biopsies inutiles) : {FP}  ({FP / max(TN + FP, 1) * 100:.1f}%)")
        print(f"\n  En diagnostic médical, minimiser les faux negatifs est prioritaire :")
        print(f"  un cancer manque peut etre fatal, une biopsie inutile est moins grave.")

    return cm
