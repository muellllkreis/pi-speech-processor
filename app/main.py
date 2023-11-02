import uvicorn
from fastapi import FastAPI

from app.models.audio_model import AudioSample
from app.services.audio_service import AudioService

audio_service = AudioService()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/audio")
def process_audio(audio_sample: AudioSample) -> str:
    return audio_service.transcribe(audio_sample)


def start():
    """Launched with `poetry run dev` at root level"""
    uvicorn.run("app.main:app", reload=True)
