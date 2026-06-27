from groq import Groq
from gtts import gTTS
import streamlit as st
import tempfile
import os

# ============================
# GROQ CLIENT
# ============================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ============================
# SPEECH TO TEXT
# ============================

def speech_to_text(audio_bytes):

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        with open(temp_path, "rb") as audio:

            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio,
                language="bn",
                response_format="text"
            )

        os.remove(temp_path)
        return transcription

    except Exception as e:
        st.error(f"❌ Voice Error: {e}")
        return None

# TEXT TO SPEECH

def text_to_speech(text):

    tts = gTTS(
        text=text,
        lang="bn"
    )

    output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(output.name)
    return output.name