"""
api.py — REST API
=================
Endpoints:
    GET  /          -> health check
    POST /predict   -> classifies an uploaded satellite image

Run with:
    uvicorn src.api:app --reload
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.predict import PredictionResult, predict

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Satellite Image Classifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_image(file: UploadFile) -> Image.Image:
    """
    Reads the uploaded file bytes and converts them to a PIL Image.
    Raises HTTP 400 if the file is not a valid image.
    """
    try:
        contents = file.file.read()
        return Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Returns a simple status message to confirm the API is running."""
    return {"status": "ok", "message": "Satellite Image Classifier API is running."}


@app.post("/predict", response_model=None)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts a satellite image and returns its predicted superclass.

    Returns:
        filename    : original filename from the upload
        label       : predicted class (Agriculture / Vegetation / Urban / Water)
        confidence  : probability assigned to the predicted class (0–1)
        probabilities: full distribution across all four classes
    """
    image: Image.Image = _read_image(file)

    try:
        result: PredictionResult = predict(image)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "filename":      file.filename,
        "label":         result.label,
        "confidence":    result.confidence,
        "probabilities": result.probabilities,
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)