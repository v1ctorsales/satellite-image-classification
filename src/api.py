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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.predict import PredictionResult, predict

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Satellite Image Classifier", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://satellite-classification.vercel.app",
        "https://satellite-image-classification-fe.vercel.app",
    ],
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
@limiter.limit("10/minute")
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Accepts a satellite image and returns its predicted superclass.
    Rate limited to 10 requests per minute per IP.

    Returns:
        filename     : original filename from the upload
        label        : predicted class (Agriculture / Vegetation / Urban / Water)
        confidence   : probability assigned to the predicted class (0–1)
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