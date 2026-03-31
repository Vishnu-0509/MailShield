# MailShield

MailShield is a simple Flask-based email classifier that predicts whether an email is spam or ham. The app includes a trained model and vectorizer to classify incoming messages and displays results in a web interface.

## Files

- `app.py` - Flask application serving the classifier UI
- `train_model.py` - Script to train the email classifier
- `requirements.txt` - Python dependencies
- `emails.csv` - Dataset used for training
- `model.pkl` - Serialized trained model
- `vectorizer.pkl` - Serialized text vectorizer used for feature extraction
- `templates/index.html` - Web UI template
- `static/style.css` - Site styling
- `static/accuracy_chart.png` - Model performance chart

## Setup

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the environment:

   - Windows PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

   - Windows Command Prompt:
     ```cmd
     .\venv\Scripts\activate.bat
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the app

Start the Flask app:

```bash
python app.py
```

Then open a browser and go to:

```
http://127.0.0.1:5000
```

## Train the model

If you want to retrain the classifier, run:

```bash
python train_model.py
```

This script will generate updated `model.pkl` and `vectorizer.pkl` files.

## Notes

- The current repository includes pre-trained model files for immediate use.
- If you want to keep the repository clean, consider adding generated files like `model.pkl` and `vectorizer.pkl` to `.gitignore`.
