import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from collections import deque
from unidecode import unidecode

load_dotenv()

st.set_page_config(
    page_title="Mixed Language STT (Roman Urdu + English)",
    page_icon="🎤","""
👂 Mixed Language Speech-to-Text Agent (Roman Urdu + English)
"""

import html
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

# Optional Groq for context enhancement
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY not found in .env or Streamlit Secrets")
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
# 🗣️ SIMPLIFIED SPEAKER PRIORITIZER
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
# 🎙️ DEEPGRAM TRANSCRIPTION (MIXED LANGUAGE)
# ============================
def transcribe_deepgram(audio_bytes):
    params = [
        ("model", "nova-3"),
        ("language", "multi"),          # 👈 Auto‑detect Urdu/English
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
    """
    Convert Arabic‑script Urdu to Roman Urdu.
    Uses unidecode if available, otherwise removes non‑Latin characters.
    """
    if not text:
        return ""
    if UNIDECODE_AVAILABLE:
        return unidecode(text)
    else:
        # Fallback: keep only ASCII + basic punctuation
        return re.sub(r'[^a-zA-Z0-9 .,\'"?!]', '', text)

# ============================
# 🧠 CONTEXT ENHANCEMENT (Optional Groq)
# ============================
def enhance_with_groq(transcript, confidence):
    if not GROQ_AVAILABLE or not GROQ_API_KEY or confidence > 0.8:
        return transcript
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are a speech recognition expert. Correct obvious errors in this transcription, but do not invent words.

Transcription: "{transcript}"
Confidence: {confidence:.1%}

Return ONLY the corrected text, nothing else."""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are conservative about corrections."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return transcript

# ============================
# 🎛️ PROCESS AUDIO (VAD + Noise Suppression)
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
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None

# ============================
# 🖥️ STREAMLIT UI
# ============================
st.title("🎤 Mixed Language STT (Roman Urdu + English)")
st.caption("Speak naturally – Urdu, English, or mix. Output in original script + Roman Urdu.")

col1, col2 = st.columns(2)
with col1:
    use_context = st.checkbox("🧠 Use Groq context correction", value=GROQ_AVAILABLE and bool(GROQ_API_KEY))
with col2:
    show_roman = st.checkbox("📝 Show Romanized text", value=True)

audio = mic_recorder(
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop",
    just_once=True,
    format="wav",
    key="mic"
)

if st.button("🗑️ Clear"):
    st.rerun()

if audio and audio.get("bytes"):
    with st.spinner("Processing audio..."):
        processed = process_audio(audio["bytes"])
        if processed:
            with st.spinner("Transcribing..."):
                trans = transcribe_deepgram(processed["processed_bytes"])
                if trans:
                    original = trans["text"]
                    confidence = trans["confidence"]

                    # Optional context correction
                    if use_context and confidence < 0.8:
                        with st.spinner("Applying context correction..."):
                            original = enhance_with_groq(original, confidence)

                    roman = romanize_text(original) if show_roman else ""

                    st.success("✅ Transcription complete")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📝 Original Script**")
                        st.code(original, language="text")
                    if show_roman:
                        with col2:
                            st.markdown("**📝 Roman Urdu**")
                            st.code(roman, language="text")

                    st.caption(f"Confidence: {confidence:.2f}")
                else:
                    st.warning("No speech detected or transcription failed")
        else:
            st.error("Audio processing failed")
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
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

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
# 🗣️ SIMPLIFIED SPEAKER PRIORITIZER
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
# 🎙️ DEEPGRAM TRANSCRIPTION (MIXED LANGUAGE)
# ============================
def transcribe_deepgram(audio_bytes):
    """
    Use Deepgram Nova-3 with multi-language detection.
    Returns original script (Arabic for Urdu) and confidence.
    """
    params = [
        ("model", "nova-3"),
        ("language", "multi"),          # 👈 Auto-detect Urdu/English/Mixed
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
# 🌐 ROMANIZATION (Arabic Script → Roman Urdu)
# ============================
def romanize(text):
    """
    Convert Arabic-script text to Roman Urdu using unidecode.
    This is a rough transliteration; for better results you can use
    a dedicated library like indic-transliteration.
    """
    if not text:
        return ""
    # unidecode converts Arabic script to Latin (e.g., "آپ" -> "ap")
    # It's not perfect for Urdu, but works for basic Roman Urdu.
    return unidecode(text)

# ============================
# 🎛️ PROCESS AUDIO (Same as before)
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
st.title("🎤 Mixed Language STT (Roman Urdu + English)")
st.caption("Speak in Urdu, English, or mix — transcription in original script + Roman Urdu")

audio = mic_recorder(
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop",
    just_once=True,
    format="wav",
    key="mic"
)

if st.button("🗑️ Clear"):
    st.rerun()

if audio and audio.get("bytes"):
    with st.spinner("Processing..."):
        res = process_audio(audio["bytes"])
        if res:
            trans = transcribe_deepgram(res["processed_bytes"])
            if trans:
                original = trans["text"]
                roman = romanize(original)
                confidence = trans["confidence"]

                st.success("✅ Transcription complete")
                st.markdown("### 📝 Original Script (Arabic/Urdu)")
                st.code(original, language="text")

                st.markdown("### 📝 Romanized (Roman Urdu)")
                st.code(roman, language="text")

                st.caption(f"Confidence: {confidence:.2f}")
            else:
                st.warning("No speech detected or transcription failed")
