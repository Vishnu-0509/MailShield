from flask import Flask, render_template, request
import joblib
import re
import string
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Check model files
# -----------------------------
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    raise FileNotFoundError("❌ model.pkl or vectorizer.pkl not found. Run train_model.py first.")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Spam Keywords List
# -----------------------------
spam_keywords = [
    "free", "winner", "urgent", "click", "offer", "cash", "prize",
    "buy now", "claim", "limited", "reward", "congratulations",
    "selected", "money", "discount", "bonus", "win", "exclusive"
]

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Extract suspicious keywords
# -----------------------------
def extract_spam_keywords(text):
    found = []
    text_lower = text.lower()
    for word in spam_keywords:
        if word in text_lower:
            found.append(word)
    return found

# -----------------------------
# Convert score to risk level
# -----------------------------
def get_risk_level(score):
    if score >= 7:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    else:
        return "LOW"

# -----------------------------
# Save scan history
# -----------------------------
def save_scan_history(text, prediction, confidence, spam_score, risk_level, suspicious_words):
    history_file = "scan_history.csv"

    new_data = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "spam_score": spam_score,
        "risk_level": risk_level,
        "suspicious_words": ", ".join(suspicious_words)
    }])

    if os.path.exists(history_file):
        new_data.to_csv(history_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(history_file, mode='w', header=True, index=False)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    spam_score = None
    risk_level = None
    message = ""
    bulk_results = None
    suspicious_words = []
    history = []

    if request.method == "POST":
        # -----------------------------
        # Single Email Prediction
        # -----------------------------
        if "message" in request.form and request.form["message"].strip():
            message = request.form["message"]
            cleaned_message = clean_text(message)
            msg_vec = vectorizer.transform([cleaned_message])

            pred = model.predict(msg_vec)[0]
            probs = model.predict_proba(msg_vec)[0]

            spam_probability = probs[1]  # probability of spam
            confidence = round(max(probs) * 100, 2)
            spam_score = round(spam_probability * 10, 2)
            risk_level = get_risk_level(spam_score)

            prediction = "SPAM EMAIL" if pred == 1 else "LEGITIMATE EMAIL"

            # Extract suspicious words
            suspicious_words = extract_spam_keywords(message)

            # Save scan history
            save_scan_history(
                text=message,
                prediction=prediction,
                confidence=confidence,
                spam_score=spam_score,
                risk_level=risk_level,
                suspicious_words=suspicious_words
            )

        # -----------------------------
        # Bulk CSV Upload Prediction
        # -----------------------------
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename.endswith(".csv"):
                df = pd.read_csv(file)

                if "text" in df.columns:
                    df["cleaned"] = df["text"].apply(clean_text)
                    vec = vectorizer.transform(df["cleaned"])
                    preds = model.predict(vec)

                    df["Prediction"] = ["SPAM" if p == 1 else "HAM" for p in preds]
                    bulk_results = df[["text", "Prediction"]].head(20).to_dict(orient="records")

    # -----------------------------
    # Load recent scan history
    # -----------------------------
    if os.path.exists("scan_history.csv"):
        history_df = pd.read_csv("scan_history.csv")
        history = history_df.tail(5).iloc[::-1].to_dict(orient="records")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        spam_score=spam_score,
        risk_level=risk_level,
        message=message,
        bulk_results=bulk_results,
        suspicious_words=suspicious_words,
        history=history
    )

if __name__ == "__main__":
    app.run(debug=True)