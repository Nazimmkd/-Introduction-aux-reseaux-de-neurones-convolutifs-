import os
import sys
from datetime import datetime

REPERTOIRES = {
    "mnist":        "sorties/mnist",
    "cifar":        "sorties/cifar",
    "modeles":      "sorties/modeles",
    "comparaison":  "sorties/comparaison",
    "ddsm":         "sorties/ddsm",
}


def chemin(dataset, nom_fichier):
    """Retourne le chemin complet pour un fichier de sortie et crée le dossier si besoin."""
    rep = REPERTOIRES[dataset]
    os.makedirs(rep, exist_ok=True)
    return os.path.join(rep, nom_fichier)


class _Tee:
    """Écrit simultanément vers le terminal et un fichier texte."""

    def __init__(self, chemin_log):
        self._terminal = sys.stdout
        self._log      = open(chemin_log, 'w', encoding='utf-8')
        horodatage     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"=== Exécution démarrée le {horodatage} ===\n\n")
        self._log.flush()

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"\n=== Exécution terminée le {horodatage} ===\n")
        self._log.close()
        sys.stdout = self._terminal


def demarrer_log(dataset):
    """
    Redirige sys.stdout vers sorties/<dataset>/execution.txt tout en
    conservant l'affichage dans le terminal.

    Utilisation recommandée :
        log = sorties.demarrer_log("mnist")
        try:
            ...
        finally:
            log.close()
    """
    chemin_log = chemin(dataset, "execution.txt")
    tee        = _Tee(chemin_log)
    sys.stdout = tee
    print(f"[LOG] Sortie capturée dans {chemin_log}")
    return tee
