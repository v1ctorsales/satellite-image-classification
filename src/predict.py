import numpy as np
from pathlib import Path
from PIL import Image
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray

#same settings from blackbox.py
IMG_SIZE   = (64, 64)
N_BINS     = 32
HOG_PIXELS = 8
HOG_CELLS  = 2
SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]

MODEL_PATH = Path("model_outputs/blackbox/blackbox_best_model.joblib")

#load the model once when initializing the api
model = joblib.load(MODEL_PATH)


def extract_color_histogram(img_array: np.ndarray) -> np.ndarray:
    features = []
    for ch in range(3):
        hist, _ = np.histogram(img_array[:, :, ch], bins=N_BINS, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.append(hist)
    return np.concatenate(features)


def extract_hog_features(img_array: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_array)
    return hog(
        gray,
        orientations=8,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        feature_vector=True,
    )


def predict_image(image: Image.Image) -> dict:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)

    features = np.concatenate([
        extract_color_histogram(arr),
        extract_hog_features(arr)
    ]).reshape(1, -1)

    label_idx   = model.predict(features)[0]
    proba       = model.predict_proba(features)[0]
    confidence  = float(proba[label_idx])
    label       = SUPER_CLASSES[label_idx]

    return {"label": label, "confidence": round(confidence, 4)}