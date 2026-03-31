from flask import Flask, render_template, request
import joblib
import re
import string
import os
import pandas as pd

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
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    message = ""
    bulk_results = None

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
            confidence = round(max(probs) * 100, 2)

            prediction = "SPAM EMAIL" if pred == 1 else "LEGITIMATE EMAIL"

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

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message,
        bulk_results=bulk_results
    )

if __name__ == "__main__":
    app.run(debug=True)