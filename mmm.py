"""
Urdu AI Agent – STT + Contextual Correction via Groq
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

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Urdu AI Agent",
    page_icon="🤖",
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
# 🧠 VAD (ADAPTIVE)
# ============================
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.speech_floor_db = 30
        self.vad_history = deque(maxlen=15)
        self.min_speech_frames = 3

    def is_speech(self, audio_chunk, current_noise_floor=None):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        energy_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        zcr = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk) + 1e-10)
        
        if current_noise_floor is not None:
            speech_threshold = current_noise_floor + 8.0
        else:
            speech_threshold = self.speech_floor_db
        
        speech = (energy_db > speech_threshold) and (energy_db > -45) and (0.01 < zcr < 0.6)
        self.vad_history.append(speech)
        if len(self.vad_history) > self.min_speech_frames:
            return sum(self.vad_history) / len(self.vad_history) > 0.5
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
        tolerance = self.primary_energy_profile * 0.6
        return abs(rms - self.primary_energy_profile) < tolerance

# ============================
# 🔊 ADAPTIVE NOISE GATE
# ============================
class AdaptiveNoiseGate:
    def __init__(self, sample_rate=16000):
        self.noise_floor = 25
        self.noise_history = deque(maxlen=50)
        self.adapt_rate = 0.1

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
        ("language", "ur"),
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
        
        transcript = alt.get("transcript", "").strip()
        confidence = float(alt.get("confidence", 0.0))
        words = alt.get("words", [])
        
        if confidence < 0.30:
            return {
                "text": "[audio unclear]",
                "confidence": confidence,
                "words": words
            }
        
        # Basic word filter (keep only confident words)
        if words:
            filtered = []
            for w in words:
                word = w.get("word", "").strip()
                wc = float(w.get("confidence", 0.0))
                if wc < 0.15:
                    filtered.append("[inaudible]")
                else:
                    filtered.append(word)
            if filtered:
                transcript = " ".join(filtered)
                transcript = re.sub(r'(\[inaudible\]\s*)+', '[inaudible]', transcript).strip()
                if not transcript or transcript == "[inaudible]":
                    return {
                        "text": "[audio unclear]",
                        "confidence": confidence,
                        "words": words
                    }
        
        return {
            "text": transcript,
            "confidence": confidence,
            "words": words
        }
    except Exception:
        return None

# ============================
# 🤖 AGENT CORRECTION VIA GROQ (THE PROMPT YOU ASKED FOR)
# ============================
def agent_correct(transcript, target_script="roman"):
    """
    Use Groq LLM to correct the transcript and convert to target script.
    This is the 'brain' of the agent.
    """
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return transcript  # fallback to raw
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Build prompt based on target script
        if target_script == "roman":
            output_format = "Roman Urdu (Latin script), without diacritics."
        else:
            output_format = "Urdu script (Arabic script)."
        
        system_prompt = f"""You are an AI agent specialized in Urdu speech recognition correction.
        Your task:
        1. The user spoke in Urdu. The raw transcription may contain errors (misheard words, missing words, extra words).
        2. Use the context to correct the transcription to what the user most likely said.
        3. Do NOT invent new information – only correct obvious errors.
        4. Output only the corrected transcription in {output_format}.
        5. If the transcription is completely unclear, output "[audio unclear]".
        """

        user_prompt = f"Raw transcription: {transcript}\n\nCorrected transcription:"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        corrected = response.choices[0].message.content.strip()
        # If the model returns empty, keep original
        return corrected if corrected else transcript
    except Exception:
        # If anything fails, return original
        return transcript

# ============================
# 🌐 ROMANIZATION (Fallback if no Groq)
# ============================
def romanize_text(arabic_script):
    if not arabic_script:
        return ""
    if UNIDECODE_AVAILABLE:
        roman = unidecode(arabic_script)
        # fix known unidecode bug
        roman = re.sub(r'\bwh\b', 'woh', roman, flags=re.IGNORECASE)
        roman = re.sub(r'\bvoh\b', 'woh', roman, flags=re.IGNORECASE)
        if len(roman) < 3 or re.search(r'[^\w\s]', roman):
            return arabic_script
        return roman
    else:
        return re.sub(r'[^a-zA-Z0-9 .,\'"?!]', '', arabic_script)

# ============================
# 🎛️ PROCESS AUDIO (SAME AS BEFORE)
# ============================
def process_audio(audio_bytes):
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.int16)
        vad = VoiceActivityDetector(sr)
        speaker = SpeakerPrioritizer()
        noise_gate = AdaptiveNoiseGate(sr)
        frame_size = sr // 50
        processed = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            noise_gate.update_noise_floor(frame)
            current_noise_floor = noise_gate.noise_floor
            is_speech = vad.is_speech(frame, current_noise_floor)
            if is_speech:
                speaker.learn_speaker(frame)
                if speaker.is_primary_speaker(frame):
                    suppressed = noise_gate.suppress(frame)
                    rms = np.sqrt(np.mean(suppressed.astype(np.float64) ** 2) + 1e-10)
                    if rms > 80:
                        processed.append(suppressed)
                    else:
                        processed.append(frame)
            else:
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
st.title("🤖 Urdu AI Agent")
st.caption("Transcribes + Corrects using Groq LLM – real intelligence.")

use_agent = st.checkbox("🧠 Enable Agent Correction (Groq)", value=GROQ_AVAILABLE and bool(GROQ_API_KEY))
show_roman = st.checkbox("Show Romanized text", value=True)
target_script = "roman" if show_roman else "arabic"

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
    with st.spinner("Processing audio..."):
        processed = process_audio(audio["bytes"])
        if processed:
            with st.spinner("Transcribing..."):
                trans = transcribe_deepgram(processed["processed_bytes"])
                if trans:
                    raw = trans["text"]
                    confidence = trans["confidence"]

                    # Apply agent correction if enabled
                    if use_agent and raw != "[audio unclear]":
                        with st.spinner("🤖 Agent correcting transcription..."):
                            corrected = agent_correct(raw, target_script)
                    else:
                        corrected = raw

                    # Romanization (if not already done by agent)
                    if show_roman and (not use_agent or target_script == "roman"):
                        final_roman = romanize_text(corrected) if corrected != "[audio unclear]" else ""
                        display_text = final_roman if final_roman else corrected
                    else:
                        display_text = corrected

                    st.success("Done")
                    st.write("**Raw Transcription:**")
                    st.code(raw, language="text")
                    st.write("**Final Output:**")
                    st.code(display_text, language="text")
                    st.caption(f"Confidence: {confidence:.2f}")
                else:
                    st.warning("No clear speech detected.")
