import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
from keras.models import load_model
from keras.layers import Layer
import tempfile
import os

# ========== Custom Attention Layer ==========
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.math.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

# ========== Load Model and Tools ==========
@st.cache_resource
def load_model_and_tools():
    model = load_model("models/trained_model.h5", custom_objects={"AttentionLayer": AttentionLayer})
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder

# ========== Feature Extraction ==========
def extract_mfcc_feature(file_path, duration=3, offset=0.5, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# ========== Streamlit UI ==========
st.set_page_config(page_title="Audio Emotion Recognition", layout="centered")
st.title("🎙️ Audio Emotion Recognition")
st.write("Upload a `.wav` file and get the predicted emotion.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract features
    features = extract_mfcc_feature(tmp_path)
    if features is not None:
        model, scaler, label_encoder = load_model_and_tools()

        X_input = scaler.transform([features]).reshape(-1, 40, 1)
        y_pred_probs = model.predict(X_input)
        y_pred_class = np.argmax(y_pred_probs, axis=1)
        predicted_label = label_encoder.inverse_transform(y_pred_class)[0]

        st.success(f"🧠 Predicted Emotion: **{predicted_label.capitalize()}**")
        st.bar_chart(y_pred_probs[0])

    os.remove(tmp_path)
