import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import pickle
from sklearn.exceptions import NotFittedError

# ---------------------------------------------------
# Load Trained Model + Label Encoder
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("modelForPrediction.sav", "rb"))
        label_encoder = pickle.load(open("label_encoder.sav", "rb"))
        return model, label_encoder
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None, None

model, label_encoder = load_model()

# ---------------------------------------------------
# Feature Extraction
# ---------------------------------------------------
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if len(X) < 2048:
                return None

            result = np.array([])
            if chroma:
                stft = np.abs(librosa.stft(X))

            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))

            if chroma:
                chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_feat))

            if mel:
                mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel_feat))

            return result
    except Exception as e:
        st.error(f"⚠️ Error extracting features: {e}")
        return None

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognition")
st.markdown("Upload a `.wav` file and get the predicted **emotion** from your trained model.")

uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    features = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)

    if features is None:
        st.error("❌ Feature extraction failed. Please upload a valid `.wav` file.")
    else:
        features = features.reshape(1, -1)
        try:
            prediction = model.predict(features)
            predicted_emotion = label_encoder.inverse_transform(prediction)[0]

            st.success(f"🎯 Predicted Emotion: **{predicted_emotion.upper()}**")

            emotion_emoji = {
                "calm": "😌", "happy": "😄", "fearful": "😨",
                "disgust": "🤢", "angry": "😠", "neutral": "😐", "sad": "😢"
            }
            if predicted_emotion in emotion_emoji:
                st.markdown(f"### {emotion_emoji[predicted_emotion]}")
        except NotFittedError:
            st.error("⚠️ The model is not fitted. Please retrain your model and save it again.")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
