
import os
import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model


BASE_DIR = r"C:\Users\HP\Desktop\job"
MODEL_DIR = os.path.join(BASE_DIR, "model")
CNN_SAVE = os.path.join(MODEL_DIR, "cnn_feature_extractor.h5")
RF_SAVE = os.path.join(MODEL_DIR, "rf_model.pkl")
LE_SAVE = os.path.join(MODEL_DIR, "label_encoder.pkl")

IMG_SIZE = 128

st.set_page_config(page_title="Palm-Astro Hybrid", layout="wide")
st.title("Palm-Astro: CNN + RandomForest (raw images)")



if not os.path.exists(CNN_SAVE) or not os.path.exists(RF_SAVE) or not os.path.exists(LE_SAVE):
    st.error("Models not found. Run train_model.py first to produce model artifacts in the 'model/' folder.")
    st.stop()

feature_model = load_model(CNN_SAVE)
rf = joblib.load(RF_SAVE)
le = joblib.load(LE_SAVE)

st.write("Upload a palm image. CNN extracts image features → RandomForest predicts dominant palm line.")



uploaded_file = st.file_uploader("Choose a palm image", type=["png", "jpg", "jpeg"])

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

    # extract CNN features
    feats = feature_model.predict(np.expand_dims(img_input, axis=0))[0]

    # random forest prediction
    pred_label = rf.predict([feats])[0]
    st.success(f"**Predicted Dominant Line:** {pred_label}")

    if hasattr(rf, "predict_proba"):
        probs = rf.predict_proba([feats])[0]
        st.write("**Confidence:**")
        for cls, p in zip(rf.classes_, probs):
            st.write(f"- {cls}: {p:.2f}")



st.header(" Palm-Astro Assistant (Local Chatbot)")

def palm_chatbot(question):
    q = question.lower()

    if "cnn" in q or "feature extractor" in q or "convolution" in q:
        return (
            "The Convolutional Neural Network (CNN) is responsible for extracting a 128-dimensional "
            "feature vector from the palm image. It captures line curves, texture, edges, and palm structure. "
            "These features are then used by the RandomForest classifier for prediction."
        )

    if "random forest" in q or "rf" in q:
        return (
            "The RandomForest classifier receives the CNN-generated features and performs the final classification. "
            "It is lightweight, fast, and allows the model to run without needing a GPU."
        )

    if "how it works" in q or "working" in q or "pipeline" in q:
        return (
            "Palm-Astro operates through a 2-stage hybrid pipeline:\n"
            "1️ The CNN extracts numerical feature vectors from palm images.\n"
            "2️ RandomForest uses those vectors to identify the dominant palm line.\n"
            "This hybrid architecture ensures accuracy while maintaining high performance even on low-end hardware."
        )


    if "dataset" in q or "data" in q:
        return (
            "The dataset is self-created and small-scale, developed manually by "
            "Mr. Nitish Kumar Sharma. It includes labeled palm images categorized into "
            "Life Line, Head Line, and Heart Line. Increasing dataset size would improve overall accuracy."
        )

    if "train" in q or "training" in q:
        return (
            "The training pipeline includes:\n"
            "- Loading the custom palm dataset\n"
            "- Training the CNN for image feature extraction\n"
            "- Extracting the CNN feature vectors for all training images\n"
            "- Training a RandomForest classifier using the extracted features\n"
            "- Saving the trained models inside the model/ directory"
        )

    if "accuracy" in q or "score" in q or "performance" in q:
        return (
            "The current model’s accuracy depends on the dataset size. Because the dataset is small and self-created, "
            "the accuracy can be improved further by adding more labeled palm images. "
            "However, the hybrid CNN+RF approach ensures stable performance even with limited data."
        )


    if "palm" in q or "life line" in q or "head line" in q or "heart line" in q or "meaning" in q:
        return (
            "The model recognizes three palmistry line categories:\n"
            "- Life Line\n"
            "- Head Line\n"
            "- Heart Line\n"
            "These labels originate from your custom dataset prepared by Mr. Nitish Kumar Sharma."
        )

    if "who made" in q or "developer" in q or "built by" in q or "creator" in q:
        return (
            "This complete project — dataset creation, annotation, CNN training, feature extraction, "
            "RandomForest model development, and Streamlit application — was designed and developed by "
            "Mr. Nitish Kumar Sharma."
        )

    if "which model" in q or "what model" in q or "model used" in q or "ai model" in q or "ml model" in q:
        return (
            "This project uses a hybrid Machine Learning + Deep Learning architecture:\n"
            "1️ A Convolutional Neural Network (CNN) is used for feature extraction.\n"
            "2️ A RandomForest classifier makes the final prediction.\n"
            "This combination is lightweight, fast, and works well even with small custom datasets."
        )

 
    if "hello" in q or "hi" in q or "hey" in q:
        return "Hello. How may I assist you regarding the Palm-Astro system?"

    if "who are you" in q:
        return (
            "I am the Palm-Astro Assistant — a local rule-based chatbot integrated inside your Streamlit application."
        )

    return (
        "I could not understand that query. You may ask about the model, dataset, CNN, RandomForest, training process, or palmistry labels."
    )


user_question = st.text_input("Ask something about the code or model:")
if user_question:
    answer = palm_chatbot(user_question)
    st.write("Answer:")
    st.write(answer)
