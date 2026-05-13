import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray

warnings.filterwarnings("ignore")

# ── Configuration
PREPROCESSING_DIR = Path(__file__).parent.parent / "preprocessing_outputs"
OUTPUT_DIR        = Path(__file__).parent.parent / "model_outputs/whitebox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE      = (64, 64)          # resize all images
N_BINS        = 32                # histogram bins per channel
HOG_PIXELS    = 8                 # pixels per HOG cell
HOG_CELLS     = 2                 # cells per block
SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]

# ── Feature extraction 

def extract_color_histogram(img_array: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Normalized color histogram for each RGB channel."""
    features = []
    for ch in range(3):
        hist, _ = np.histogram(img_array[:, :, ch], bins=n_bins, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.append(hist)
    return np.concatenate(features)


def extract_hog_features(img_array: np.ndarray) -> np.ndarray:
    """HOG (Histogram of Oriented Gradients) in grayscale."""
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
    """Loads image and returns concatenated feature vector."""
    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img)
        color_feat = extract_color_histogram(arr)
        hog_feat   = extract_hog_features(arr)
        return np.concatenate([color_feat, hog_feat])
    except Exception as e:
        print(f"  [WARNING] Error processing {image_path}: {e}")
        return None


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Reads CSV, extracts features and returns (X, y)."""
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

# ── Training ────

def train():
    t0 = time.time()

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("validation")
    X_test,  y_test  = load_split("test")

    # Pipeline: normalization + decision tree
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            max_depth=20,           # limits depth to avoid overfitting
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("\n[TRAINING] Fitting Decision Tree...")
    model.fit(X_train, y_train)

    # ── Evaluation ────
    results = {}
    for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        acc    = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=SUPER_CLASSES, output_dict=True)
        cm     = confusion_matrix(y, y_pred).tolist()
        results[split_name] = {"accuracy": acc, "report": report, "confusion_matrix": cm}

        print(f"\n── {split_name.upper()} ──")
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y, y_pred, target_names=SUPER_CLASSES))

    elapsed = time.time() - t0
    results["training_time_s"] = round(elapsed, 2)
    results["feature_dim"]     = X_train.shape[1]
    results["n_train"]         = len(X_train)
    results["model"]           = "DecisionTree"

    # ── Save ─────
    joblib.dump(model, OUTPUT_DIR / "whitebox_model.joblib")
    print(f"\n[SAVED] Model → {OUTPUT_DIR / 'whitebox_model.joblib'}")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Metrics → {OUTPUT_DIR / 'metrics.json'}")

    # Export tree as text (readable = white box!)
    tree_text = export_text(
        model.named_steps["clf"],
        feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
        max_depth=5,  # show only the first 5 levels
    )
    with open(OUTPUT_DIR / "tree_structure.txt", "w") as f:
        f.write(tree_text)
    print(f"[SAVED] Tree structure (top-5 levels) → {OUTPUT_DIR / 'tree_structure.txt'}")

    print(f"\n✅ Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    train()