import os
import numpy as np
import librosa
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer
import keras.backend as K
from sklearn.metrics import classification_report

# ===== Fix: Custom Attention Layer =====
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
        # Use TensorFlow ops directly
        e = tf.math.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

# ===== Configuration =====
TEST_DIRS = ["path to test directory"]
MODEL_PATH = "models/trained_model.h5"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

# ===== Helper Functions =====
def parse_emotion(file_name):
    emotion_dict = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    code = file_name.split("-")[2]
    return emotion_dict.get(code, "unknown")

def gather_file_paths(root_dirs):
    collected = []
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    full_path = os.path.join(root, file)
                    label = parse_emotion(file)
                    collected.append((full_path, label))
    return pd.DataFrame(collected, columns=["filepath", "emotion"])

def extract_mfcc_feature(path, duration=3, offset=0.5, n_mfcc=40):
    try:
        y, sr = librosa.load(path, duration=duration, offset=offset)
        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc_feat.T, axis=0)
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return None

# ===== Load Test Data =====
print("Loading test data...")
test_df = gather_file_paths(TEST_DIRS)
test_df.dropna(inplace=True)

X_test = []
y_true = []

for _, row in test_df.iterrows():
    feature = extract_mfcc_feature(row['filepath'])
    if feature is not None:
        X_test.append(feature)
        y_true.append(row['emotion'])

X_test = np.array(X_test)

# ===== Load Tools =====
print("Loading model and tools...")
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled.reshape(-1, 40, 1)

model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})

# ===== Predict and Evaluate =====
print("Evaluating model...")
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Optional: check class distribution
print("Test class distribution:", np.unique(y_true, return_counts=True))

# Evaluation report
print("\nClassification Report:\n")
# Convert numeric predictions/ground truth to string labels
# y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Print the report
print(classification_report(
    y_true, y_pred_labels, target_names=label_encoder.classes_, zero_division=0))


