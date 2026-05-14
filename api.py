from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io

from src.predict import predict_image, predict_image_whitebox

app = FastAPI(title="Satellite API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/sendImage")
async def send_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_image(image)
    return {"filename": file.filename, **result}

@app.post("/sendImage/whitebox")
async def send_image_whitebox(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_image_whitebox(image)
    return {"filename": file.filename, **result}

@app.post("/sendImage/compare")
async def send_image_compare(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    blackbox = predict_image(image)
    whitebox = predict_image_whitebox(image)
    return {
        "filename": file.filename,
        "blackbox": blackbox,
        "whitebox": whitebox,
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)