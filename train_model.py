import pandas as pd
import re
import string
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

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
# Check if file exists
# -----------------------------
if not os.path.exists("emails.csv"):
    print("❌ ERROR: emails.csv file not found!")
    exit()

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("emails.csv", encoding="latin-1")
print("📂 Dataset loaded successfully!")
print("Columns found:", df.columns.tolist())

if 'text' not in df.columns or 'spam' not in df.columns:
    print("❌ ERROR: emails.csv must contain columns 'text' and 'spam'")
    exit()

df = df[['text', 'spam']]
df.columns = ['message', 'label']
df.dropna(inplace=True)
df['message'] = df['message'].apply(clean_text)

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n✅ Accuracy:", accuracy)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model and Vectorizer
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")

# -----------------------------
# Create Accuracy Chart
# -----------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.tight_layout()

os.makedirs("static", exist_ok=True)
plt.savefig("static/accuracy_chart.png")
print("📊 Accuracy chart saved as static/accuracy_chart.png")