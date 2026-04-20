from src.data_preprocessing import clean

from flask import Flask, request, jsonify
from datetime import datetime
import joblib

# Load model
model = joblib.load("../models/spam_classifier.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    spam_example = "Don't miss our new AMAZING FEATURE by entering your credit card data!"
    ham_example = "Dear Hiring Manager, Please find attached my application. Sincerely, Me."

    html = f""" <h1>Mail Classifier Spam/Ham</h1>
    <p>Enter your text below:</p>
    <form action="/predict" method="get">
        <textarea name="text" rows="10" cols="80"></textarea>
        <br><br>
        <button type="submit">PREDICT!</button>
    </form>
    <br><br>
    <p> If you're not feeling inspired don't hesitate to use these examples : </p>
    <ul><li>{spam_example}</li><li>{ham_example}</li></ul>
    """
    return html


@app.route("/predict", methods=["GET"])
def predict_api():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Missing text parameter"}), 400
    cleaned_text = clean(text)
    prediction = model.predict([cleaned_text])[0]
    proba = model.predict_proba([cleaned_text]).max()
    return jsonify({"text": text, "cleaned_text": cleaned_text, "prediction": prediction,
                    "confidence": round(float(proba), 2), "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(debug=True)