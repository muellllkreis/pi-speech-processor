import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/audio")
def process_audio(audio_data: bytes):
    return {"Hello": "World"}


def start():
    """Launched with `poetry run dev` at root level"""
    uvicorn.run("app.main:app", reload=True)
