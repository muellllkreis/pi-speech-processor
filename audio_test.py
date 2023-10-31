import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Initialize audio recording parameters
duration = 30  # seconds
sampling_rate = 16000  # 16kHz
audio_data = np.zeros((sampling_rate * duration,))
stop_recording = False


def record_audio(audio_data, stream, duration, sampling_rate):
    start_idx = 0
    for _ in range(duration):
        audio_chunk, overflowed = stream.read(sampling_rate)
        stop_idx = start_idx + audio_chunk.shape[0]
        if stop_idx > audio_data.shape[0]:
            break
        audio_data[start_idx:stop_idx] = audio_chunk.squeeze()
        start_idx = stop_idx
        if stop_recording:
            break


# Set up the recording stream
with sd.InputStream(samplerate=sampling_rate, channels=1) as stream:
    print("Press Enter to start recording...")
    input()

    print(f"Recording for up to {duration} seconds. Press Enter to stop early...")

    # Start recording in a separate thread
    recording_thread = threading.Thread(
        target=record_audio, args=(audio_data, stream, duration, sampling_rate)
    )
    recording_thread.start()

    # Wait for Enter to be pressed to stop the recording
    input()
    stop_recording = True
    recording_thread.join()

    print("Done!")

# Truncate zeros if stopped early
audio_data = audio_data[: np.max(np.nonzero(audio_data)) + 1]
sf.write("debug_audio.wav", audio_data, sampling_rate)

# Process audio features
input_features = processor(
    audio_data, sampling_rate=sampling_rate, return_tensors="pt"
).input_features

# Generate token IDs
predicted_ids = model.generate(input_features)

# Decode transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
