"""
predict.py — Inference module
==============================
Loads the trained model lazily (only on first use) and exposes a single
public function:

    predict(image: Image.Image) -> PredictionResult

The model is never loaded at import time — this prevents the API from
crashing on startup if the model file does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

# ── Paths & constants ─────────────────────────────────────────────────────────

MODEL_PATH    = Path(__file__).parent.parent / "outputs" / "model" / "model.joblib"
SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]
IMG_SIZE      = (64, 64)
N_BINS        = 32
HOG_PIXELS    = 8
HOG_CELLS     = 2

# ── Lazy model loader ─────────────────────────────────────────────────────────

_model = None   # module-level cache; populated on first call to predict()


def _load_model():
    """
    Loads the model from disk the first time it is needed.
    Raises a clear error if the file is missing, instead of crashing
    silently at import time.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run train.py first to generate it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Feature extraction (mirrors train.py exactly) ────────────────────────────

def _color_histogram(image_array: np.ndarray) -> np.ndarray:
    channel_histograms = []
    for channel in range(3):
        hist, _ = np.histogram(image_array[:, :, channel], bins=N_BINS, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        channel_histograms.append(hist)
    return np.concatenate(channel_histograms)


def _hog_features(image_array: np.ndarray) -> np.ndarray:
    grayscale = rgb2gray(image_array)
    return hog(
        grayscale,
        orientations=8,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        feature_vector=True,
    )


def _extract_features(image: Image.Image) -> np.ndarray:
    """Converts a PIL image to the feature vector expected by the model."""
    array = np.array(image.convert("RGB").resize(IMG_SIZE))
    features = np.concatenate([_color_histogram(array), _hog_features(array)])
    return features.reshape(1, -1)   # shape (1, n_features) for scikit-learn


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    label:         str
    confidence:    float
    probabilities: dict[str, float]


# ── Public API ────────────────────────────────────────────────────────────────

def predict(image: Image.Image) -> PredictionResult:
    """
    Classifies a satellite image into one of four superclasses:
        Agriculture, Vegetation, Urban, Water

    Args:
        image: PIL Image (any mode — converted to RGB internally)

    Returns:
        PredictionResult with the predicted label, confidence score,
        and full probability distribution across all classes.
    """
    model    = _load_model()
    features = _extract_features(image)

    label_idx = int(model.predict(features)[0])
    proba     = model.predict_proba(features)[0]

    return PredictionResult(
        label=SUPER_CLASSES[label_idx],
        confidence=round(float(proba[label_idx]), 4),
        probabilities={
            cls: round(float(proba[i]), 4)
            for i, cls in enumerate(SUPER_CLASSES)
        },
    )