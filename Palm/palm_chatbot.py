# -------------------------------
# Palm-Astro Prediction Chatbot
# -------------------------------

def analyze_palm_line(pred_label, confidence_dict=None):
    """
    Returns a detailed full-statement analysis based on the predicted palm line.
    """

    # Make confidence text
    conf_text = ""
    if confidence_dict:
        conf_text += "Confidence Levels:\n"
        for line, score in confidence_dict.items():
            conf_text += f"- {line}: {score:.2f}\n"
        conf_text += "\n"

    # Meaning based on predicted line
    if pred_label.lower() == "life line":
        meaning = (
            "Life Line Analysis\n\n"
            "Your Life Line indicates vitality, lifestyle energy, and overall strength. "
            "A strong prediction of Life Line dominance suggests:\n"
            "- You have good physical stamina and resilience.\n"
            "- You recover quickly from challenges.\n"
            "- You possess stability and grounded personality traits.\n\n"
            "This does NOT predict lifespan — it reflects energy, health balance, and "
            "your natural ability to overcome everyday obstacles."
        )

    elif pred_label.lower() == "head line":
        meaning = (
            "Head Line Analysis\n\n"
            "Your Head Line represents mindset, intelligence style, and decision-making. "
            "A strong Head Line prediction suggests:\n"
            "- You are an analytical thinker.\n"
            "- You prefer logic over impulsiveness.\n"
            "- Creativity and problem-solving come naturally.\n"
            "- You take time before making important decisions.\n\n"
            "This shows a sharp, disciplined, intelligent personality."
        )

    elif pred_label.lower() == "heart line":
        meaning = (
            "Heart Line Analysis\n\n"
            "Your Heart Line reflects emotions, relationships, and empathy. "
            "A dominant Heart Line prediction suggests:\n"
            "- You have strong emotional intelligence.\n"
            "- You form deep connections with people.\n"
            "- You value relationships and loyalty.\n"
            "- You may be sensitive or intuitive.\n\n"
            "This indicates a warm, caring, emotionally expressive nature."
        )

    else:
        meaning = (
            f"Prediction received: **{pred_label}**.\n"
            "However, detailed analysis is available only for Life Line, Head Line, or Heart Line."
        )

    return conf_text + meaning


# ----------------------------------------------------
# Simple info chatbot (your existing rule-based one)
# ----------------------------------------------------
def palm_info_chatbot(question):
    """
    Handles questions about model, training, CNN, RF etc.
    """
    q = question.lower()

    if "cnn" in q or "feature extractor" in q:
        return (
            "The CNN extracts a 128-dimensional feature vector from palm images. "
            "It identifies edges, curves, shapes, and textures before passing "
            "the features to the RandomForest classifier."
        )

    if "random forest" in q:
        return (
            "RandomForest takes the CNN-extracted features and performs the final classification. "
            "It is fast and works well on small datasets."
        )

    if "how it works" in q or "pipeline" in q:
        return (
            "Palm-Astro works in two stages:\n"
            "1. CNN extracts meaningful features from the palm image.\n"
            "2. RandomForest predicts Life Line / Head Line / Heart Line.\n"
        )

    if "dataset" in q:
        return (
            "The dataset is a custom palm image dataset with Life Line, Head Line, and Heart Line labels."
        )

    if "hello" in q or "hi" in q:
        return "Hello! Ask me anything about the Palm-Astro system."

    return "I couldn't understand that. Try asking about the model, CNN, RF, dataset, or working process."
