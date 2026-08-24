"""
Mixed Language STT (Roman Urdu + English)
"""
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from collections import deque

# Optional romanization
try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Mixed Language STT",
    page_icon="🎤",
    layout="wide"
)

# ============================
# 🔑 API KEYS
# ============================
def get_api_key(var_name):
    try:
        key = os.getenv(var_name)
        if key:
            return key
        return st.secrets.get(var_name, None)
    except Exception:
        return os.getenv(var_name)

DEEPGRAM_API_KEY = get_api_key("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY not found")
    st.stop()

# ============================
# 🧠 VAD (No librosa)
# ============================
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.speech_floor_db = 35
        self.vad_history = deque(maxlen=30)

    def is_speech(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        energy_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        zcr = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk) + 1e-10)
        speech = energy_db > self.speech_floor_db and 0.01 < zcr < 0.6
        self.vad_history.append(speech)
        if len(self.vad_history) > 10:
            return sum(self.vad_history) / len(self.vad_history) > 0.4
        return speech

# ============================
# 🗣️ SPEAKER PRIORITIZER
# ============================
class SpeakerPrioritizer:
    def __init__(self):
        self.primary_energy_profile = None
        self.learning_count = 0
        self.max_learning = 30

    def learn_speaker(self, audio_chunk):
        if self.learning_count >= self.max_learning:
            return
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        if self.primary_energy_profile is None:
            self.primary_energy_profile = rms
        else:
            alpha = 0.1
            self.primary_energy_profile = alpha * rms + (1 - alpha) * self.primary_energy_profile
        self.learning_count += 1

    def is_primary_speaker(self, audio_chunk):
        if self.learning_count < 5:
            return True
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        if self.primary_energy_profile is None:
            return True
        tolerance = self.primary_energy_profile * 0.5
        return abs(rms - self.primary_energy_profile) < tolerance

# ============================
# 🔊 ADAPTIVE NOISE GATE
# ============================
class AdaptiveNoiseGate:
    def __init__(self, sample_rate=16000):
        self.noise_floor = 25
        self.noise_history = deque(maxlen=50)
        self.adapt_rate = 0.05

    def update_noise_floor(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        db = 20 * np.log10(rms / 32768.0 + 1e-10)
        self.noise_history.append(db)
        if len(self.noise_history) > 20:
            new_floor = np.percentile(list(self.noise_history), 25)
            self.noise_floor = self.adapt_rate * new_floor + (1 - self.adapt_rate) * self.noise_floor

    def suppress(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        db = 20 * np.log10(rms / 32768.0 + 1e-10)
        if db < self.noise_floor + 5:
            atten = max(0.0, (self.noise_floor + 10.0 - db) / 10.0)
            return (audio_chunk.astype(np.float64) * (1.0 - atten * 0.7)).astype(np.int16)
        return audio_chunk

# ============================
# 🎙️ DEEPGRAM TRANSCRIPTION
# ============================
def transcribe_deepgram(audio_bytes):
    params = [
        ("model", "nova-3"),
        ("language", "multi"),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
    ]
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    try:
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        alt = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
        return {
            "text": alt.get("transcript", "").strip(),
            "confidence": float(alt.get("confidence", 0.0))
        }
    except Exception:
        return None

# ============================
# 🌐 ROMANIZATION
# ============================
def romanize_text(text):
    if not text:
        return ""
    if UNIDECODE_AVAILABLE:
        return unidecode(text)
    else:
        return re.sub(r'[^a-zA-Z0-9 .,\'"?!]', '', text)

# ============================
# 🎛️ PROCESS AUDIO
# ============================
def process_audio(audio_bytes):
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.int16)
        vad = VoiceActivityDetector(sr)
        speaker = SpeakerPrioritizer()
        noise = AdaptiveNoiseGate(sr)
        frame_size = sr // 50
        processed = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            if vad.is_speech(frame):
                speaker.learn_speaker(frame)
                if speaker.is_primary_speaker(frame):
                    processed.append(noise.suppress(frame))
            else:
                noise.update_noise_floor(frame)
                processed.append(frame)
        if not processed:
            return None
        out = io.BytesIO()
        wav.write(out, sr, np.concatenate(processed))
        out.seek(0)
        return {"processed_bytes": out.read()}
    except Exception:
        return None

# ============================
# 🖥️ UI
# ============================
st.title("🎤 Mixed Language STT (Urdu + English)")
st.caption("Speak naturally. Output in original script + Roman Urdu.")

show_roman = st.checkbox("Show Romanized text", value=True)

audio = mic_recorder(
    start_prompt="Start Speaking",
    stop_prompt="Stop",
    just_once=True,
    format="wav",
    key="mic"
)

if st.button("Clear"):
    st.rerun()

if audio and audio.get("bytes"):
    with st.spinner("Processing..."):
        processed = process_audio(audio["bytes"])
        if processed:
            trans = transcribe_deepgram(processed["processed_bytes"])
            if trans:
                original = trans["text"]
                roman = romanize_text(original) if show_roman else ""

                st.success("Done")
                st.write("**Original Script:**")
                st.code(original, language="text")

                if show_roman and roman:
                    st.write("**Roman Urdu:**")
                    st.code(roman, language="text")

                st.caption(f"Confidence: {trans['confidence']:.2f}")
            else:
                st.warning("No speech detected")
