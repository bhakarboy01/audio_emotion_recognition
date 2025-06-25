import streamlit as st
import librosa
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
import os
import tempfile

# Set page config
st.set_page_config(page_title="Emotion Classifier", page_icon="🎤", layout="wide")

# Load the trained model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = load_model('models\emotion_model.h5')  # Replace with your actual model path
    scaler = joblib.load('scaler.pkl')  # Replace with your actual scaler path
    le = joblib.load('label_encoder.pkl')  # Replace with your actual label encoder path
    return model, scaler, le

model, scaler, le = load_artifacts()

# Emotion mapping
emotion_map = {
    'neutral': '😐 Neutral',
    'calm': '😌 Calm',
    'happy': '😊 Happy',
    'sad': '😢 Sad',
    'angry': '😠 Angry',
    'fearful': '😨 Fearful',
    'disgust': '🤢 Disgust',
    'surprised': '😲 Surprised'
}

# Function to extract features (modified to match training preprocessing)
def extract_features(path, max_pad_len=174):
    try:
        y, sr = librosa.load(path, sr=22050)
        stft = np.abs(librosa.stft(y))

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)

        # Pad or truncate
        def pad_or_trunc(x, max_len):
            if x.shape[1] < max_len:
                return np.pad(x, ((0, 0), (0, max_len - x.shape[1])), mode='constant')
            else:
                return x[:, :max_len]

        mfcc = pad_or_trunc(mfcc, max_pad_len)
        chroma = pad_or_trunc(chroma, max_pad_len)
        mel = pad_or_trunc(mel, max_pad_len)
        contrast = pad_or_trunc(contrast, max_pad_len)

        # Stack features and flatten to match training shape
        stacked = np.vstack([mfcc, chroma, mel, contrast])
        flattened = stacked.reshape(1, -1)  # Flatten to (1, n_features)
        
        return flattened
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Streamlit app
def main():
    st.title("🎤 Audio Emotion Classifier")
    st.write("Upload an audio file (WAV format recommended) and we'll predict the emotion!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Display audio player
            st.audio(tmp_path)
            
            # Extract features
            with st.spinner("Analyzing audio features..."):
                features = extract_features(tmp_path)
                
                if features is not None:
                    # Scale features (now matches training shape)
                    X_scaled = scaler.transform(features)
                    
                    # Reshape for model input (1, 187, 174)
                    X_final = X_scaled.reshape(1, 187, 174)
                    
                    # Make prediction
                    prediction = model.predict(X_final)
                    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
                    confidence = np.max(prediction)
                    
                    # Display results
                    st.success("Analysis complete!")
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Emotion", emotion_map.get(predicted_label, predicted_label))
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Display probabilities for all emotions
                    st.subheader("Emotion Probabilities")
                    probs = prediction[0]
                    sorted_indices = np.argsort(probs)[::-1]
                    
                    for i in sorted_indices:
                        emotion = le.inverse_transform([i])[0]
                        prob = probs[i]
                        st.progress(float(prob), text=f"{emotion_map.get(emotion, emotion)}: {prob*100:.2f}%")
                    
                    # Display the most likely emotion with an emoji
                    st.balloons()
                    st.success(f"The audio sounds {emotion_map.get(predicted_label, predicted_label)}!")
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()