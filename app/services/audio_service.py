from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app.models.audio_model import AudioSample


class AudioService:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    def transcribe(self, audio_sample: AudioSample) -> str:
        input_features = self.processor(
            audio_sample.audio_data, sampling_rate=audio_sample.sampling_rate, return_tensors="pt"
        ).input_features

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0].strip()
