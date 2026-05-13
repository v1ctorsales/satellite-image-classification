import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef
from skimage.feature import hog
from skimage.color import rgb2gray

warnings.filterwarnings("ignore")

# ── Configuration
PREPROCESSING_DIR = Path(__file__).parent.parent / "preprocessing_outputs"
OUTPUT_DIR        = Path(__file__).parent.parent / "model_outputs/blackbox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE      = (64, 64)
N_BINS        = 32
HOG_PIXELS    = 8
HOG_CELLS     = 2
SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]

# ── Feature extraction

def extract_color_histogram(img_array: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    features = []
    for ch in range(3):
        hist, _ = np.histogram(img_array[:, :, ch], bins=n_bins, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.append(hist)
    return np.concatenate(features)


def extract_hog_features(img_array: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_array)
    feat = hog(
        gray,
        orientations=8,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        feature_vector=True,
    )
    return feat


def extract_features(image_path: str) -> np.ndarray | None:
    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img)
        return np.concatenate([extract_color_histogram(arr), extract_hog_features(arr)])
    except Exception as e:
        print(f"  [WARNING] Error processing {image_path}: {e}")
        return None


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = PREPROCESSING_DIR / f"{split}_remapped.csv"
    df = pd.read_csv(csv_path)
    print(f"\n[{split.upper()}] Loading {len(df)} images...")

    X_list, y_list = [], []
    for _, row in df.iterrows():
        feat = extract_features(row["ImagePath"])
        if feat is not None:
            X_list.append(feat)
            y_list.append(int(row["Label"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  Features: {X.shape}  |  Classes: {np.bincount(y).tolist()}")
    return X, y

# ── Model definitions

def build_models() -> dict:
    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,            # uses all CPU cores
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }

    # LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            )),
        ])
        print("[INFO] LightGBM detected and included in comparison.")
    except ImportError:
        print("[INFO] LightGBM not installed — using only Random Forest.")
        print("       To install: pip install lightgbm")

    return models

# ── Training and comparison

def train():
    t0 = time.time()

    # ── Diagnóstico: verifica se há imagens repetidas entre splits
    print("\n[DIAGNOSTICO] Verificando sobreposição entre splits...")
    splits_paths = {}
    for split in ["train", "validation", "test"]:
        csv_path = PREPROCESSING_DIR / f"{split}_remapped.csv"
        df_check = pd.read_csv(csv_path)
        splits_paths[split] = set(df_check["ImagePath"].tolist())

    train_val = splits_paths["train"] & splits_paths["validation"]
    train_test = splits_paths["train"] & splits_paths["test"]
    val_test = splits_paths["validation"] & splits_paths["test"]

    print(f"  Train ∩ Validation : {len(train_val)} imagens em comum")
    print(f"  Train ∩ Test       : {len(train_test)} imagens em comum")
    print(f"  Validation ∩ Test  : {len(val_test)} imagens em comum")
    if train_val or train_test:
        print("  ⚠️  LEAKAGE DETECTADO — imagens do treino presentes na avaliação!")
    else:
        print("  ✅ Sem sobreposição — splits limpos")

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("validation")
    X_test,  y_test  = load_split("test")

    models  = build_models()
    results = {}
    best_acc, best_name, best_model = 0.0, None, None

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"[TRAINING] {name}...")
        t_start = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t_start

        model_results = {"training_time_s": round(t_fit, 2), "model": name}

        for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            mcc = matthews_corrcoef(y, y_pred)
            report = classification_report(y, y_pred, target_names=SUPER_CLASSES, output_dict=True)
            cm = confusion_matrix(y, y_pred).tolist()
            model_results[split_name] = {
                "accuracy": acc, "mcc": mcc, "report": report, "confusion_matrix": cm
            }
            print(f"\n  [{split_name.upper()}] Accuracy: {acc:.4f}  |  MCC: {mcc:.4f}")
            print(classification_report(y, y_pred, target_names=SUPER_CLASSES))

        results[name] = model_results

        val_acc = model_results["validation"]["accuracy"]
        if val_acc > best_acc:
            best_acc, best_name, best_model = val_acc, name, model

    # ── Save best model
    print(f"\n{'═'*50}")
    print(f" Best model: {best_name}  (val accuracy: {best_acc:.4f})")

    joblib.dump(best_model, OUTPUT_DIR / "blackbox_best_model.joblib")
    print(f"[SAVED] Model → {OUTPUT_DIR / 'blackbox_best_model.joblib'}")

    results["best_model"] = best_name
    results["total_time_s"] = round(time.time() - t0, 2)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Metrics → {OUTPUT_DIR / 'metrics.json'}")

    # Feature importance of the best model (if available)
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        top_n = 20
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
        importance_dict = {
            f"feat_{i}": float(importances[i]) for i in top_idx
        }
        with open(OUTPUT_DIR / "feature_importance_top20.json", "w") as f:
            json.dump(importance_dict, f, indent=2)
        print(f"[SAVED] Top-20 features → {OUTPUT_DIR / 'feature_importance_top20.json'}")

    print(f"\n✅ Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    train()