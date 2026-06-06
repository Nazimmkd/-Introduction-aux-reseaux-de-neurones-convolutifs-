import numpy as np
import matplotlib.pyplot as plt

CIFAR_CLASSES = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
                 'chien', 'grenouille', 'cheval', 'bateau', 'camion']


def visualiser_exemples(X, y, titre="Exemples", class_names=None,
                        image_shape=(28, 28), cmap='gray'):
    """
    Affiche un exemple par classe (10 images au total).
    X : (D, N) pour MNIST  ou  (N, H, W, C) pour CIFAR
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(titre, fontsize=16)
    for i in range(10):
        idx = np.where(y == i)[0][0]
        ax  = axes[i // 5, i % 5]
        if X.ndim == 2:
            ax.imshow(X[:, idx].reshape(image_shape), cmap=cmap)
        else:
            ax.imshow(X[idx])
        ax.set_title(class_names[i] if class_names else str(i))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualiser_erreurs(X, y, predictions, titre="Exemples mal classés",
                       class_names=None, image_shape=(28, 28), cmap='gray',
                       save_path=None):
    """
    Affiche les 10 premières erreurs de classification.
    X : (D, N) pour MNIST  ou  (N, H, W, C) pour CIFAR
    """
    error_idx = np.where(predictions != y)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(titre, fontsize=13)

    for i, idx in enumerate(error_idx[:10]):
        ax = axes[i // 5][i % 5]
        if X.ndim == 2:
            ax.imshow(X[:, idx].reshape(image_shape), cmap=cmap)
            vrai   = str(y[idx])
            predit = str(predictions[idx])
        else:
            ax.imshow(X[idx])
            vrai   = class_names[y[idx]]   if class_names else str(y[idx])
            predit = class_names[predictions[idx]] if class_names else str(predictions[idx])
        ax.set_title(f"Vrai:{vrai}\nPrédit:{predit}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualiser_erreurs_communes(X, y, pred_dict, image_shape=(28, 28), save_path=None):
    """
    Affiche les exemples mal classés par TOUS les modèles du dictionnaire.
    pred_dict : {"Linéaire": pred_lin, "MLP H=1": pred_h1, ...}
    """
    preds    = list(pred_dict.values())
    communes = np.where(np.all([p != y for p in preds], axis=0))[0]
    noms     = list(pred_dict.keys())
    print(f"\nExemples mal classés par les {len(noms)} modèles : {len(communes)}")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"Erreurs communes ({', '.join(noms)})", fontsize=13)
    for i, idx in enumerate(communes[:10]):
        ax = axes[i // 5][i % 5]
        ax.imshow(X[:, idx].reshape(image_shape), cmap='gray')
        lignes = " ".join([f"{n[0]}:{pred_dict[n][idx]}" for n in noms])
        ax.set_title(f"Vrai:{y[idx]}\n{lignes}", fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def tracer_convergence(histories, labels, titre="Convergence (Log Loss)", save_path=None):
    """
    Trace les courbes de perte par époque.
    histories : liste de listes (une par modèle) — les listes vides sont ignorées
    labels    : liste de noms correspondants
    """
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        if hist:
            plt.plot(hist, label=label)
    plt.title(titre)
    plt.xlabel("Époques")
    plt.ylabel("Erreur (Log Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def tracer_acp(X_2d, y, titre="Visualisation ACP", class_names=None, save_path=None):
    """
    Scatter plot des données projetées en 2D par ACP.
    X_2d : (N, 2)
    """
    plt.figure(figsize=(10, 8))
    if class_names:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i in range(10):
            mask = y == i
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[colors[i]], label=class_names[i], alpha=0.6, s=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=5, alpha=0.5)
        plt.colorbar(scatter, label="Classes (0-9)")
    plt.title(titre)
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ── Visualisations de comparaison ─────────────────────────────

def tracer_comparaison_barres(resultats, activations, architectures,
                               titre="Comparaison", save_path=None):
    """
    Graphique en barres groupées : activations × architectures.
    resultats : dict { (activation, arch_nom): accuracy }
    """
    arch_noms = list(architectures.keys())
    x         = np.arange(len(arch_noms))
    n_act     = len(activations)
    largeur   = 0.8 / n_act

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, activation in enumerate(activations):
        vals     = [resultats.get((activation, arch), 0) * 100 for arch in arch_noms]
        decalage = x + i * largeur - 0.4 + largeur / 2
        bars     = ax.bar(decalage, vals, largeur, label=activation)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha='center', va='bottom', fontsize=7)

    ax.set_xlabel("Architecture")
    ax.set_ylabel("Précision test (%)")
    ax.set_title(titre)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_noms, rotation=20, ha='right')
    ax.legend(title="Activation")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def tracer_confusion(cm, classes, titre="Matrice de confusion", save_path=None):
    """
    Heatmap annotée d'une matrice de confusion.
    cm      : numpy (n_classes, n_classes)
    classes : liste de noms de classes
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel",   fontsize=12)
    ax.set_title(titre, fontsize=13)

    total = max(cm.sum(), 1)
    seuil = cm.max() * 0.6
    for i in range(len(classes)):
        for j in range(len(classes)):
            couleur = 'white' if cm[i, j] > seuil else 'black'
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)",
                    ha='center', va='center', color=couleur,
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def tracer_heatmap_resultats(resultats, activations, arch_noms,
                              titre="Heatmap précision", save_path=None):
    """
    Heatmap : lignes = activations, colonnes = architectures.
    resultats : dict { (activation, arch_nom): accuracy }
    """
    data = np.array([
        [resultats.get((act, arch), 0) * 100 for arch in arch_noms]
        for act in activations
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(arch_noms) * 2), max(4, len(activations) * 1.2)))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(arch_noms)))
    ax.set_xticklabels(arch_noms, rotation=30, ha='right')
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels(activations)

    for i in range(len(activations)):
        for j in range(len(arch_noms)):
            couleur = 'white' if data[i, j] > 65 else 'black'
            ax.text(j, i, f"{data[i, j]:.1f}%", ha='center', va='center',
                    color=couleur, fontsize=9, fontweight='bold')

    plt.colorbar(im, label="Précision (%)")
    ax.set_title(titre)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
