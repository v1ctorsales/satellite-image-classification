from fastapi import FastAPI
import uvicorn

#uvicorn api:app --reload

app = FastAPI(title="Satellite API", version="1.0.0")

@app.get("/")
def root():
    return {"message": "API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)