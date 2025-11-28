import os
import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# IMPORT NEW CHATBOT MODULE
from palm_chatbot import analyze_palm_line, palm_info_chatbot


# -------------------------
# FIXED: Correct model path
# -------------------------
BASE_DIR = r"C:\Users\HP\Desktop\nww\palm_m\Palm"
MODEL_DIR = os.path.join(BASE_DIR, "model")
CNN_SAVE = os.path.join(MODEL_DIR, "cnn_feature_extractor.h5")
RF_SAVE = os.path.join(MODEL_DIR, "rf_model.pkl")
LE_SAVE = os.path.join(MODEL_DIR, "label_encoder.pkl")

IMG_SIZE = 128

st.set_page_config(page_title="Palm-Astro Hybrid", layout="wide")
st.title("Palm-Astro: CNN + RandomForest (raw images)")


# -------------------------
# Check model existence
# -------------------------
if not os.path.exists(CNN_SAVE) or not os.path.exists(RF_SAVE) or not os.path.exists(LE_SAVE):
    st.error("Models not found. Run train_model.py first to produce model artifacts in the 'model/' folder.")
    st.stop()

feature_model = load_model(CNN_SAVE)
rf = joblib.load(RF_SAVE)
le = joblib.load(LE_SAVE)

st.write("Upload a palm image. CNN extracts image features → RandomForest predicts dominant palm line.")


# -------------------------
# Upload + Predict
# -------------------------
uploaded_file = st.file_uploader("Choose a palm image", type=["png", "jpg", "jpeg"])

prediction_done = False
final_pred_label = None
final_confidence = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the image.")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.astype("float32") / 255.0

    st.image(img_rgb, caption="Uploaded Palm Image", use_container_width=True)

    # Extract features
    feats = feature_model.predict(np.expand_dims(img_input, axis=0))[0]

    # Predict
    pred_label = rf.predict([feats])[0]
    prediction_done = True
    final_pred_label = pred_label

    st.success(f"**Predicted Dominant Line:** {pred_label}")

    # Confidence
    confidence_dict = {}
    if hasattr(rf, "predict_proba"):
        probs = rf.predict_proba([feats])[0]
        st.write("**Confidence:**")
        for cls, p in zip(rf.classes_, probs):
            st.write(f"- {cls}: {p:.2f}")
            confidence_dict[cls] = p

    final_confidence = confidence_dict

    # -------------------------
    # SHOW PALM READING ANALYSIS
    # -------------------------
    st.subheader("🔮 Full Palm Line Interpretation")
    analysis_text = analyze_palm_line(final_pred_label, final_confidence)
    st.write(analysis_text)


# -------------------------
# Local Chatbot
# -------------------------
st.header("💬 Palm-Astro Assistant (Local Chatbot)")

user_question = st.text_input("Ask something about the code, model, CNN, RF, dataset, or pipeline:")

if user_question:
    answer = palm_info_chatbot(user_question)
    st.write("Answer:")
    st.write(answer)
