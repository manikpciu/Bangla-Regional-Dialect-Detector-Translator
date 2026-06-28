import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from voice import speech_to_text, text_to_speech
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import os
import json
import joblib
from difflib import get_close_matches
from pipeline import RegionalLexiconFeatureExtractor
from gtts import gTTS


# ============================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="DialectBridge",
    page_icon="🗣️",
    layout="centered",
)


# GROQ SETUP

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# LOAD ML ARTIFACTS

@st.cache_resource
def load_artifacts():

    feature_pipeline = joblib.load(
        "models/full_feature_pipeline.joblib"
    )

    model = joblib.load(
        "models/stacking_ensemble_model_5cv.joblib"
    )

    label_encoder = joblib.load(
        "models/label_encoder.joblib"
    )

    return feature_pipeline, model, label_encoder


feature_pipeline, model, label_encoder = load_artifacts()

# ============================================================
@st.cache_resource
def load_translations():

    def _load(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {

        "Chittagong": _load("translations/chittagong.json"),

        "Nowoakhali": _load("translations/noakhali.json"),

        "Noakhali": _load("translations/noakhali.json"),

        "Barishal": _load("translations/barishal.json"),

        "Rangpur": _load("translations/rangpur.json")

    }

translations = load_translations()


# ============================================================
# HELPER : SMART DICTIONARY SEARCH
# ============================================================

def smart_translate(user_text: str, dictionary: dict) -> str | None:
    """Try exact match → punctuation-stripped match → fuzzy match."""

    # 1. Exact match
    if user_text in dictionary:
        return dictionary[user_text]

    # 2. Strip leading/trailing whitespace
    cleaned = user_text.strip()
    if cleaned in dictionary:
        return dictionary[cleaned]

    # 3. Strip common punctuation
    for ch in "।.?!,":
        cleaned = cleaned.replace(ch, "")

    def _strip(key: str) -> str:
        for ch in "।.?!,":
            key = key.replace(ch, "")
        return key.strip()

    for key, value in dictionary.items():
        if cleaned == _strip(key):
            return value

    # 4. Fuzzy match
    match = get_close_matches(cleaned, dictionary.keys(), n=1, cutoff=0.65)
    if match:
        return dictionary[match[0]]

    return None


# ============================================================
# HELPER : GEMINI FALLBACK TRANSLATION
# ============================================================

def groq_translate(text: str, region: str) -> str | None:

    prompt = f"""তুমি বাংলাদেশের ভাষাবিদ।
নিচের বাক্যটি {region} অঞ্চলের আঞ্চলিক ভাষায় লেখা।
এটিকে প্রমিত বাংলায় অনুবাদ করো।
শুধু অনুবাদ দিবে, আর কিছু না।

বাক্য: {text}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"AI Translation Error: {e}")
        return None

# ============================================================
# HELPER : TEXT-TO-SPEECH
# ============================================================

def generate_audio(text: str, output_path: str = "output/output.mp3") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang="bn")
    tts.save(output_path)
    return output_path


# ============================================================
# SESSION STATE : translation history
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ---- Main button ---- */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #145a86);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        height: 55px;
        width: 100%;
        border: none;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ---- Text area ---- */
    .stTextArea textarea {
        font-size: 17px;
        border-radius: 10px;
    }

    /* ---- Alert boxes ---- */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---- Code block (copy box) ---- */
    .stCode {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🗣️ DialectBridge")
st.markdown("### Bangla Regional Dialect → Standard Bangla")
st.write(
    "Detect regional dialects and translate them into Standard Bangla using AI."
)
st.markdown("---")


# ============================================================
# EXAMPLE DROPDOWN  (top, before text area)
# ============================================================

examples = [
    "মুই মোর ভাষা নিয়ে খুশি",
    "চিটাঙ্গ আঁরার জান",
    "ক্যামোন আচেন বাহেরা?",
    "ও বাবু, আর কদিন এন্নে চইলবো ?",
]

sample = st.selectbox("📌 Try an Example", ["— Select Example —"] + examples)

# Pre-fill text area if user picked an example
default_text = sample if sample != "— Select Example —" else ""

st.markdown("## ✍️ Input")

tab1, tab2 = st.tabs([
    "⌨️ Text",
    "🎤 Voice"
])

user_text = ""

with tab1:

    user_text = st.text_area(
    "Enter Dialect Sentence",
    value=default_text,  
    height=150,
)
    
with tab2:

    st.write("Press the microphone and speak.")

    audio = mic_recorder(

        start_prompt="🎤 Start Recording",

        stop_prompt="⏹ Stop Recording",

        key="recorder"

    )

if audio:

    # Audio Preview
    st.audio(audio["bytes"])

    # Speech → Text
    with st.spinner("🎤 Converting voice to text..."):

        user_text = speech_to_text(
            audio["bytes"]
        )

    # Voice এসেছে এটা Detect Section-কে জানাবে
    st.session_state.voice_input = True

    # Success Message
    st.success("✅ Voice converted successfully.")

    # Recognized Text
    st.text_area(

        "📝 Recognized Text",

        value=user_text,

        height=150,

        disabled=True

    )

    # Auto Translate Message
    st.info("🤖 Voice detected. Processing translation automatically...")

# ============================================================
# ============================================================
# DETECT & TRANSLATE
# ============================================================

process = False

# Text Mode
if st.button("🚀 Detect & Translate"):
    process = True

# Voice Mode (Auto Process)
if user_text and st.session_state.get("voice_input", False):
    process = True
    st.session_state.voice_input = False


if process:

    if not user_text.strip():

        st.warning("Please enter a dialect sentence.")

    else:

        try:

            # ======================================
            # Feature Extraction
            # ======================================

            X = feature_pipeline.transform([user_text])

            # ======================================
            # Prediction
            # ======================================

            prediction = model.predict(X)

            region = label_encoder.inverse_transform(
                prediction
            )[0]

            # ======================================
            # Confidence
            # ======================================

            confidence = None

            if hasattr(model, "predict_proba"):

                try:

                    confidence = (
                        model.predict_proba(X).max() * 100
                    )

                except Exception:

                    confidence = None

            # ======================================
            # Region Output
            # ======================================

            st.success(
                f"📍 Detected Region : {region}"
            )

            if confidence is not None:

                st.info(
                    f"🎯 Confidence : {confidence:.2f}%"
                )

            # ======================================
            # Dictionary Translation
            # ======================================

            dictionary = translations.get(
                region,
                {}
            )

            standard = smart_translate(
                user_text,
                dictionary
            )

            source = "Dictionary"

            # ======================================
            # Groq AI Fallback
            # ======================================

            if standard is None:

                with st.spinner(
                    "🤖 Groq AI is translating..."
                ):

                    standard = groq_translate(
                        user_text,
                        region
                    )

                source = "Groq AI"

            # ======================================
            # Translation Output
            # ======================================

            if standard:

                st.markdown("---")

                st.subheader("📖 Standard Bangla")

                st.success(standard)

                if source == "Dictionary":

                    st.caption(
                        "✅ Source : Translation Dictionary"
                    )

                else:

                    st.caption(
                        "🤖 Source : Groq AI (Llama-3.3-70B)"
                    )

                # ======================================
                # Audio
                # ======================================

                st.markdown("---")

                st.subheader("🔊 Audio Output")

                audio_path = text_to_speech(
                    standard
                )

                with open(audio_path, "rb") as audio_file:

                    st.audio(
                        audio_file.read(),
                        format="audio/mp3"
                    )

                # ======================================
                # Copy
                # ======================================

                st.markdown("---")

                st.subheader("📋 Copy Translation")

                st.code(
                    standard,
                    language=None
                )

                # ======================================
                # Download
                # ======================================

                st.download_button(

                    label="⬇ Download Translation",

                    data=standard,

                    file_name="translation.txt",

                    mime="text/plain"

                )

                # ======================================
                # History
                # ======================================

                st.session_state.history.append({

                    "Dialect": user_text,

                    "Region": region,

                    "Standard": standard,

                    "Source": source

                })

            else:

                st.error(

                    "❌ Translation could not be generated."

                )

        except Exception as e:

            st.error(
                f"❌ Error : {e}"
            )

# TRANSLATION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:

    st.markdown("---")

    st.subheader("📜 Translation History")

    for i, item in enumerate(reversed(st.session_state.history), start=1):

        with st.expander(f"{i}. {item['Dialect']}"):

            st.write("🌍 **Detected Region**")
            st.info(item["Region"])

            st.write("📖 **Standard Bangla**")
            st.success(item["Standard"])

            st.write("🤖 **Translation Source**")
            st.caption(item["Source"])