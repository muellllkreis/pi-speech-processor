from pydantic import BaseModel


class AudioSample(BaseModel):
    audio_data: list[float]
    sampling_rate: int
