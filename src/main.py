from data_preprocessing import get_data, data_cleaning, split_data
from modeling import train_model, evaluate_model

# Get dataset and clean it
df = data_cleaning(get_data())
# Train test split
X_train, X_test, y_train, y_test = split_data(df)
# Vectorizer and training
model = train_model(X_train, y_train)
# Evaluation
evaluate_model(model, X_test, y_test)