# 🗣️ DialectBridge
### AI-Powered Bangla Regional Dialect Detection & Translation System

DialectBridge is a machine learning web application that automatically detects regional Bangla dialects and translates them into Standard Bangla using a Stacking Ensemble Model and Google Gemini AI.

---

## 🌍 Supported Regions
- Chittagong (চট্টগ্রাম)
- Noakhali (নোয়াখালী)
- Barishal (বরিশাল)
- Rangpur (রংপুর)

---

## ✨ Features
- 🔍 **Dialect Detection** — Identifies the regional origin of a Bangla sentence
- 🎯 **Confidence Score** — Shows model prediction confidence in percentage
- 📖 **Dictionary Translation** — Fast lookup from curated regional dictionaries
- 🤖 **Gemini AI Fallback** — Google Gemini translates sentences not found in dictionary
- 🔊 **Speech Output** — Converts Standard Bangla translation to audio
- 📋 **Copy & Download** — Export translation as a `.txt` file
- 📜 **Translation History** — Tracks all translations within the session

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | Stacking Ensemble (scikit-learn) |
| AI Fallback | Google Gemini 2.5 Flash |
| Text-to-Speech | gTTS |
| Language | Python 3.11 |

---

## 📁 Project Structure
dialect/

├── app.py

├── pipeline.py

├── create_json.py

├── requirements.txt

├── models/

│   ├── full_feature_pipeline.joblib

│   ├── stacking_ensemble_model_5cv.joblib

│   └── label_encoder.joblib

├── translations/

│   ├── chittagong.json

│   ├── noakhali.json

│   ├── barishal.json

│   └── rangpur.json

└── dataset/

└── dialect_translation.csv
