import numpy as np
import os


DOSSIER_DDSM = "CBIS-DDSM"


def _construire_index_jpeg(dossier_jpeg):
    """Construit {uid_dossier: chemin_jpg} en scannant le dossier jpeg/."""
    index = {}
    for uid in os.listdir(dossier_jpeg):
        chemin_uid = os.path.join(dossier_jpeg, uid)
        if not os.path.isdir(chemin_uid):
            continue
        for f in sorted(os.listdir(chemin_uid)):
            if f.lower().endswith('.jpg'):
                index[uid] = os.path.join(chemin_uid, f)
                break
    return index


def _trouver_jpeg(chemin_csv, index_jpeg):
    """
    Essaie chaque composante UID du chemin CSV (format DICOM) contre l'index JPEG.
    Chemin CSV typique : "Mass-Training_P_00001_LEFT_CC/<UID1>/<UID2>/000000.dcm"
    """
    for partie in chemin_csv.strip().replace('\n', '').replace('\r', '').split('/'):
        if partie in index_jpeg:
            return index_jpeg[partie]
    return None


def _charger_split(csv_path, index_jpeg, taille, col_image='image file path'):
    import csv
    from PIL import Image

    X, y = [], []
    vu, sauts = set(), 0

    with open(csv_path, encoding='utf-8') as f:
        for ligne in csv.DictReader(f):
            cle = ligne[col_image].strip()
            if cle in vu:
                continue
            vu.add(cle)

            chemin_jpg = _trouver_jpeg(cle, index_jpeg)
            if chemin_jpg is None:
                sauts += 1
                continue

            try:
                img = Image.open(chemin_jpg).convert('L')
                img = img.resize((taille, taille), Image.LANCZOS)
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(1 if ligne['pathology'].strip() == 'MALIGNANT' else 0)
            except Exception:
                sauts += 1

    if sauts:
        print(f"  {sauts} image(s) ignoree(s) (fichier manquant ou invalide)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def charger_cbis_ddsm(taille=128, dossier=DOSSIER_DDSM, utiliser_crop=True):
    """
    Charge les mammographies CBIS-DDSM pour la classification binaire bénin/malin.

    utiliser_crop=True  : crop centré sur la masse (recommandé).
                          Préserve la texture discriminante malgré le redimensionnement.
    utiliser_crop=False : mammographie complète (résolution ~4000px réduite à taille px).

    Retourne X_train (N,H,W), y_train, X_test (N,H,W), y_test en float32.
    Cache dans data_models/ pour éviter de relire les JPEG à chaque exécution.
    """
    os.makedirs("data_models", exist_ok=True)
    prefixe = "crop" if utiliser_crop else "full"
    tag     = f"ddsm_{prefixe}_{taille}"
    col_img = 'cropped image file path' if utiliser_crop else 'image file path'
    chemins = {k: os.path.join("data_models", f"{k}_{tag}.npy")
               for k in ("X_train", "y_train", "X_test", "y_test")}

    if all(os.path.exists(p) for p in chemins.values()):
        print(f"[LOAD] CBIS-DDSM {prefixe} ({taille}x{taille}) — chargement depuis cache...")
        X_train = np.load(chemins["X_train"])
        y_train = np.load(chemins["y_train"])
        X_test  = np.load(chemins["X_test"])
        y_test  = np.load(chemins["y_test"])
    else:
        print(f"[PREPROCESS] CBIS-DDSM {prefixe} — redimensionnement vers {taille}x{taille}...")
        index = _construire_index_jpeg(os.path.join(dossier, "jpeg"))
        print(f"  {len(index)} dossiers JPEG indexes")

        csv_dir = os.path.join(dossier, "csv")
        X_train, y_train = _charger_split(
            os.path.join(csv_dir, "mass_case_description_train_set.csv"), index, taille, col_img)
        X_test, y_test = _charger_split(
            os.path.join(csv_dir, "mass_case_description_test_set.csv"),  index, taille, col_img)

        np.save(chemins["X_train"], X_train)
        np.save(chemins["y_train"], y_train)
        np.save(chemins["X_test"],  X_test)
        np.save(chemins["y_test"],  y_test)
        print(f"  Cache sauvegarde dans data_models/")

    n_mal_tr = int((y_train == 1).sum())
    n_mal_te = int((y_test  == 1).sum())
    print(f"\n  Distribution — Train : {len(y_train)} images "
          f"({len(y_train) - n_mal_tr} benins, {n_mal_tr} malins)")
    print(f"              Test  : {len(y_test)}  images "
          f"({len(y_test) - n_mal_te} benins, {n_mal_te} malins)")
    taux = n_mal_tr / max(len(y_train), 1) * 100
    print(f"  Taux de malignite (train) : {taux:.1f}%")

    return X_train, y_train, X_test, y_test
