# Audio Models - Speech & Sound AI

## Overview

**Audio Models** process and generate audio signals including speech, music, and environmental sounds. This category encompasses Speech Recognition (STT), Text-to-Speech (TTS), Music Generation, Audio Classification, and Sound Event Detection.

## Categories

| Category | Input | Output | Examples |
|----------|-------|--------|----------|
| **ASR/STT** | Audio | Text | Whisper, Deepgram |
| **TTS** | Text | Audio | ElevenLabs, Bark |
| **Voice Cloning** | Audio + Text | Audio | XTTS, RVC |
| **Music Generation** | Text/Audio | Music | MusicGen, Suno |
| **Audio Classification** | Audio | Labels | YAMNet, PANNs |
| **Sound Separation** | Mixed Audio | Separated Audio | Demucs |
| **Audio Editing** | Audio + Text | Audio | AudioCraft |

---

## Speech Recognition (ASR/STT)

### Key Models

| Model | Provider | Languages | Best For |
|-------|----------|-----------|----------|
| **Whisper** | OpenAI | 100+ | General transcription |
| **Whisper Large-v3** | OpenAI | 100+ | Highest accuracy |
| **Deepgram Nova-2** | Deepgram | 36+ | Real-time, accuracy |
| **Assembly AI** | AssemblyAI | 40+ | Speaker diarization |
| **Faster-Whisper** | Community | 100+ | Fast inference |
| **Seamless** | Meta | 100+ | Translation |
| **Canary** | NVIDIA | 10+ | Enterprise |

### Example: Whisper Transcription

```python
import whisper
import torch

# Load model
model = whisper.load_model("large-v3")

# Transcribe
result = model.transcribe(
    "audio.mp3",
    language="en",  # Optional: auto-detect if None
    task="transcribe"  # or "translate" to English
)

print(result["text"])

# With timestamps
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
```

### Example: OpenAI API

```python
from openai import OpenAI

client = OpenAI()

# Transcribe
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

print(transcription.text)

# With word timestamps
for word in transcription.words:
    print(f"{word.start:.2f}s: {word.word}")
```

### Example: Real-time with Faster-Whisper

```python
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

# GPU-optimized model
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Transcribe with streaming
segments, info = model.transcribe(
    "audio.wav",
    beam_size=5,
    vad_filter=True,  # Voice activity detection
    vad_parameters=dict(min_silence_duration_ms=500)
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Example: Speaker Diarization

```python
from pyannote.audio import Pipeline
import whisper

# Load diarization pipeline (requires HuggingFace token)
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# Run diarization
diarization_result = diarization("audio.wav")

# Combine with Whisper
whisper_model = whisper.load_model("large-v3")
transcription = whisper_model.transcribe("audio.wav")

# Match segments to speakers
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

---

## Text-to-Speech (TTS)

### Key Models

| Model | Provider | Quality | Speed | Voice Cloning |
|-------|----------|---------|-------|---------------|
| **ElevenLabs** | ElevenLabs | Excellent | Fast | ✅ Yes |
| **OpenAI TTS** | OpenAI | Very Good | Fast | ❌ No |
| **Bark** | Suno | Good | Slow | ❌ No |
| **XTTS** | Coqui | Excellent | Medium | ✅ Yes |
| **Tortoise** | Community | Excellent | Very Slow | ✅ Yes |
| **Piper** | Rhasspy | Good | Very Fast | ❌ No |
| **Coqui TTS** | Coqui | Good | Medium | ✅ Yes |
| **StyleTTS2** | Community | Excellent | Medium | ✅ Yes |

### Example: OpenAI TTS

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI()

speech_file_path = Path("output.mp3")

response = client.audio.speech.create(
    model="tts-1-hd",  # or "tts-1" for faster
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! This is a test of OpenAI's text-to-speech.",
    speed=1.0  # 0.25 to 4.0
)

response.stream_to_file(speech_file_path)
```

### Example: ElevenLabs

```python
from elevenlabs import generate, play, save, voices
from elevenlabs import set_api_key

set_api_key("YOUR_API_KEY")

# Generate speech
audio = generate(
    text="Hello world! This is ElevenLabs.",
    voice="Rachel",  # or voice ID
    model="eleven_multilingual_v2"
)

# Play or save
play(audio)
save(audio, "output.mp3")

# List available voices
available_voices = voices()
for voice in available_voices:
    print(f"{voice.name}: {voice.voice_id}")
```

### Example: Bark (Open Source, Expressive)

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Load models
preload_models()

# Generate with emotions and non-speech sounds
text = """
Hello! [laughs] This is Bark.
It can add... [sighs] various expressions.
And even ♪ sing a little tune ♪
"""

audio_array = generate_audio(text, history_prompt="v2/en_speaker_6")

# Save
write_wav("bark_output.wav", SAMPLE_RATE, audio_array)
```

### Example: XTTS Voice Cloning

```python
from TTS.api import TTS
import torch

# Initialize with voice cloning model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Clone voice from reference audio
tts.tts_to_file(
    text="Hello, this is my cloned voice speaking.",
    speaker_wav="reference_voice.wav",  # 6+ seconds of clean audio
    language="en",
    file_path="cloned_output.wav"
)

# Multi-speaker with cloned voices
speakers = {
    "Alice": "alice_voice.wav",
    "Bob": "bob_voice.wav"
}

for name, ref_audio in speakers.items():
    tts.tts_to_file(
        text=f"Hello, I am {name}.",
        speaker_wav=ref_audio,
        language="en",
        file_path=f"{name}_output.wav"
    )
```

---

## Music Generation

### Key Models

| Model | Provider | Input | Output | Best For |
|-------|----------|-------|--------|----------|
| **Suno v3** | Suno | Text | Full songs | Complete tracks |
| **MusicGen** | Meta | Text/Melody | Music | Instrumentals |
| **Stable Audio** | Stability | Text | Audio | Sound design |
| **Riffusion** | Community | Text | Music | Quick demos |
| **AudioCraft** | Meta | Text | Audio | Research |
| **Udio** | Udio | Text | Songs | Vocals + music |

### Example: MusicGen

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=30)  # 30 seconds

# Generate from text
descriptions = [
    "Upbeat electronic dance music with heavy bass",
    "Gentle acoustic guitar folk song",
    "Epic orchestral cinematic score"
]

wav = model.generate(descriptions)

# Save
for idx, one_wav in enumerate(wav):
    audio_write(
        f'output_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )

# Generate continuation/variation
melody_audio, sr = torchaudio.load("melody.wav")
wav = model.generate_with_chroma(
    descriptions=["Jazz version of this melody"],
    melody_wavs=melody_audio,
    melody_sample_rate=sr
)
```

### Example: Suno API

```python
import requests

# Suno requires authentication and their API
# This is a simplified example

def generate_song(prompt, style):
    response = requests.post(
        "https://api.suno.ai/v1/generate",
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={
            "prompt": prompt,
            "style": style,  # e.g., "pop", "rock", "classical"
            "duration": 120  # seconds
        }
    )
    return response.json()

result = generate_song(
    prompt="A song about coding late at night",
    style="synthwave"
)
```

---

## Audio Classification

### Key Models

| Model | Classes | Best For |
|-------|---------|----------|
| **YAMNet** | 521 | Environmental sounds |
| **PANNs** | 527 | AudioSet classification |
| **Audio Spectrogram Transformer** | Various | General |
| **Wav2Vec2** | Custom | Fine-tuning |
| **CLAP** | Zero-shot | Text-audio matching |

### Example: Audio Classification

```python
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load model
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Load audio
waveform, sample_rate = torchaudio.load("audio.wav")

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Process
inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# Get predictions
predicted_class = logits.argmax(-1).item()
label = model.config.id2label[predicted_class]
print(f"Predicted: {label}")
```

### Example: CLAP (Zero-shot Audio Classification)

```python
from transformers import ClapProcessor, ClapModel
import torchaudio

# Load model
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# Load audio
waveform, sample_rate = torchaudio.load("audio.wav")

# Define candidate labels
labels = ["dog barking", "music playing", "people talking", "car horn"]

# Process
inputs = processor(
    audios=waveform.squeeze().numpy(),
    sampling_rate=sample_rate,
    text=labels,
    return_tensors="pt",
    padding=True
)

# Get similarities
outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio
probs = logits_per_audio.softmax(dim=-1)

for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")
```

---

## Audio Separation & Enhancement

### Key Models

| Model | Task | Best For |
|-------|------|----------|
| **Demucs** | Source separation | Music stems |
| **Spleeter** | Vocal separation | Quick separation |
| **RNNoise** | Noise reduction | Voice cleanup |
| **DeepFilterNet** | Enhancement | Real-time |
| **Resemble Enhance** | Enhancement | Voice quality |

### Example: Demucs (Stem Separation)

```python
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model

# Load model
model = pretrained.get_model("htdemucs")
model.eval()

# Load audio
waveform, sample_rate = torchaudio.load("song.mp3")

# Separate
sources = apply_model(
    model,
    waveform.unsqueeze(0),
    device="cuda"
)

# sources shape: (batch, stems, channels, samples)
# stems: drums, bass, other, vocals

stem_names = ["drums", "bass", "other", "vocals"]
for idx, name in enumerate(stem_names):
    torchaudio.save(f"{name}.wav", sources[0, idx], sample_rate)
```

### Example: Audio Enhancement

```python
import torch
import torchaudio
from df import enhance, init_df

# Initialize DeepFilterNet
model, df_state, _ = init_df()

# Load noisy audio
noisy, sample_rate = torchaudio.load("noisy_audio.wav")

# Enhance
enhanced = enhance(model, df_state, noisy)

# Save
torchaudio.save("enhanced_audio.wav", enhanced, sample_rate)
```

---

## Comparison: ASR Models

| Model | WER (LibriSpeech) | Speed | Languages | Real-time |
|-------|-------------------|-------|-----------|-----------|
| Whisper Large-v3 | 2.7% | 0.2x | 100+ | ❌ |
| Faster-Whisper Large | 2.7% | 4x | 100+ | ✅ |
| Deepgram Nova-2 | 3.1% | Real-time | 36 | ✅ |
| AssemblyAI | 3.8% | Real-time | 40 | ✅ |
| wav2vec2 | 4.5% | Fast | Fine-tuned | ✅ |

## Comparison: TTS Models

| Model | Naturalness | Latency | Languages | Voice Cloning |
|-------|-------------|---------|-----------|---------------|
| ElevenLabs | 9/10 | Low | 29+ | ✅ |
| XTTS | 8/10 | Medium | 17 | ✅ |
| OpenAI TTS | 8/10 | Low | 50+ | ❌ |
| Bark | 7/10 | High | 13+ | ❌ |
| Piper | 6/10 | Very Low | 30+ | ❌ |

## Choosing Audio Models

```
What's your task?
│
├── Speech Recognition
│   ├── Accuracy priority → Whisper Large-v3
│   ├── Real-time needed → Deepgram Nova-2
│   ├── Self-hosted → Faster-Whisper
│   └── Speaker separation → AssemblyAI
│
├── Text-to-Speech
│   ├── Best quality → ElevenLabs
│   ├── Voice cloning → XTTS or ElevenLabs
│   ├── Fast/cheap → OpenAI TTS or Piper
│   └── Expressive/emotions → Bark
│
├── Music Generation
│   ├── Full songs with vocals → Suno
│   ├── Instrumentals → MusicGen
│   └── Sound effects → Stable Audio
│
├── Audio Classification
│   ├── Fixed classes → AST or PANNs
│   └── Zero-shot → CLAP
│
└── Audio Processing
    ├── Stem separation → Demucs
    └── Noise reduction → DeepFilterNet
```

## Related Models

- **[LLM](./LLM.md)** - Combined with audio for voice assistants
- **[LMM](./LMM.md)** - Multimodal including audio
- **[DIFFUSION](./DIFFUSION.md)** - Audio diffusion models

## Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Bark](https://github.com/suno-ai/bark)
- [AudioCraft/MusicGen](https://github.com/facebookresearch/audiocraft)
- [Demucs](https://github.com/facebookresearch/demucs)
- [ElevenLabs](https://elevenlabs.io/)
- [Suno](https://suno.ai/)
