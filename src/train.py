"""
train.py — Satellite Image Classifier (Random Forest)
======================================================
Steps:
    1. check_data_leakage()  -> ensures splits do not overlap
    2. load_dataset()        -> loads images and extracts features
    3. build_model()         -> defines the training pipeline
    4. evaluate()            -> computes metrics on a split
    5. save_outputs()        -> persists model and metrics to disk
    6. train()               -> orchestrates everything in order
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

PREPROCESSING_DIR = Path(__file__).parent.parent / "outputs" / "preprocessing"
OUTPUT_DIR        = Path(__file__).parent.parent / "outputs" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE      = (64, 64)
N_BINS        = 32
HOG_PIXELS    = 8
HOG_CELLS     = 2
SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_data_leakage():
    """
    Ensures no training image appears in the validation or test splits.
    Leakage inflates metrics and invalidates results.
    """
    paths_by_split = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(PREPROCESSING_DIR / f"{split}_remapped.csv")
        paths_by_split[split] = set(df["ImagePath"])

    train_paths = paths_by_split["train"]
    overlaps = {
        "train x validation": train_paths & paths_by_split["validation"],
        "train x test":       train_paths & paths_by_split["test"],
    }

    print("[LEAKAGE CHECK]")
    for pair, overlap in overlaps.items():
        status = "LEAKAGE DETECTED!" if overlap else "OK"
        print(f"  {pair}: {len(overlap)} images in common — {status}")

    if any(overlaps.values()):
        raise RuntimeError("Data leakage detected. Fix the splits before continuing.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def color_histogram(image_array: np.ndarray) -> np.ndarray:
    """
    Returns a normalized color histogram for each RGB channel.
    Output: vector of size N_BINS * 3 = 96 values.
    """
    channel_histograms = []
    for channel in range(3):
        hist, _ = np.histogram(image_array[:, :, channel], bins=N_BINS, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        channel_histograms.append(hist)
    return np.concatenate(channel_histograms)


def hog_features(image_array: np.ndarray) -> np.ndarray:
    """
    Returns HOG (Histogram of Oriented Gradients) features.
    HOG captures edges and textures — useful for distinguishing
    urban areas from vegetation.
    """
    grayscale = rgb2gray(image_array)
    return hog(
        grayscale,
        orientations=8,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        feature_vector=True,
    )


def extract_features(image_path: str) -> np.ndarray | None:
    """
    Loads an image, resizes it, and returns the full feature vector:
        [histogram_R | histogram_G | histogram_B | HOG]
    Returns None if the image cannot be loaded.
    """
    try:
        image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        array = np.array(image)
        return np.concatenate([color_histogram(array), hog_features(array)])
    except Exception as e:
        print(f"  [WARNING] Could not process {image_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the CSV for a given split (train/validation/test), extracts
    features from each image, and returns (X, y) ready for scikit-learn.
    """
    csv_path = PREPROCESSING_DIR / f"{split}_remapped.csv"
    df = pd.read_csv(csv_path)
    print(f"\n[{split.upper()}] Processing {len(df)} images...")

    features_list, labels_list = [], []
    for _, row in df.iterrows():
        features = extract_features(row["ImagePath"])
        if features is not None:
            features_list.append(features)
            labels_list.append(int(row["Label"]))

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list,   dtype=np.int32)
    print(f"  X shape: {X.shape} | Samples per class: {np.bincount(y).tolist()}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    Returns a scikit-learn Pipeline with two steps:
        1. StandardScaler  -> normalizes features (mean 0, std 1)
        2. RandomForest    -> ensemble of 300 decision trees
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=1,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    """
    Generates predictions and computes accuracy, MCC, per-class F1,
    and confusion matrix.
    MCC (Matthews Correlation Coefficient) is more robust than accuracy
    on imbalanced datasets — ranges from -1 (worst) to +1 (perfect).
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy":         accuracy_score(y, y_pred),
        "mcc":              matthews_corrcoef(y, y_pred),
        "report":           classification_report(
                                y, y_pred,
                                target_names=SUPER_CLASSES,
                                output_dict=True,
                            ),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    print(f"\n── {split_name.upper()} ──")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(classification_report(y, y_pred, target_names=SUPER_CLASSES))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(model: Pipeline, results: dict):
    """Saves the trained model and metrics to disk."""
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    print(f"\n[SAVED] Model             -> {OUTPUT_DIR / 'model.joblib'}")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Metrics           -> {OUTPUT_DIR / 'metrics.json'}")

    importances = model.named_steps["classifier"].feature_importances_
    top_indices  = np.argsort(importances)[::-1][:20]
    top_features = {f"feat_{i}": float(importances[i]) for i in top_indices}

    with open(OUTPUT_DIR / "feature_importance.json", "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"[SAVED] Feature importance -> {OUTPUT_DIR / 'feature_importance.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def train():
    t_start = time.time()

    check_data_leakage()

    X_train, y_train = load_dataset("train")
    X_val,   y_val   = load_dataset("validation")
    X_test,  y_test  = load_dataset("test")

    model = build_model()
    print("\n[TRAINING] Fitting Random Forest...")
    model.fit(X_train, y_train)

    results = {
        "model":           "RandomForest",
        "validation":      evaluate(model, X_val,  y_val,  "validation"),
        "test":            evaluate(model, X_test, y_test, "test"),
        "training_time_s": round(time.time() - t_start, 2),
    }

    save_outputs(model, results)
    print(f"\nDone in {results['training_time_s']:.1f}s")


if __name__ == "__main__":
    train()