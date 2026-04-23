from data_preprocessing import get_data, data_cleaning, split_data
from modeling import train_model, evaluate_model
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "spam_classifier.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Get dataset and clean it
df = data_cleaning(get_data())
# Train test split
X_train, X_test, y_train, y_test = split_data(df)
# Vectorizer and training
model = train_model(X_train, y_train)
joblib.dump(model, "../models/spam_classifier.pkl")
print("Model trained successfully")
print(f"Saved at: {MODEL_PATH}")
# Evaluation
evaluate_model(model, X_test, y_test)