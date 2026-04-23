# spam_classification
This project implements an end-to-end spam email classification system.

A machine learning model is trained on a Kaggle dataset using TF-IDF vectorization and a Naive Bayes classifier. The trained model is then exposed through a REST API, allowing users to classify new emails.

The project covers the full pipeline from data exploration and preprocessing to model training, evaluation, and deployment.

## The dataset
Dataset from Kaggle :  ([link](https://www.kaggle.com/datasets/willyard/spam-email-dataset)) 

## Project structure
```bash
.
├── data
│   ├── enron_spam_data.csv
│   └── spam_email_dataset.csv
├── LICENSE
├── models
│   └── spam_classifier.pkl
├── notebooks
│   └── dataset_exploration.ipynb
├── README.md
├── requirements.txt
├── results
├── service
│   ├── __init__.py
│   └── spam_classification_service.py
└── src
    ├── data_preprocessing.py
    ├── __init__.py
    ├── main.py
    └── modeling.py
```
The folders `data/*` and `models/*` are not included in the repository. You need to follow the installation steps before using the API.

## Installation
Clone the repository :
```bash
git clone https://github.com/anaelle-p/spam_classification
cd spam_classification
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Before using the API, you'll need to train the model.
First download the dataset from Kaggle ([link](https://www.kaggle.com/datasets/willyard/spam-email-dataset)). Create a `data/` directory where you will put the downloaded dataset csv file.
Then train the model:
```bash
python src/main.py
```
This will generate a trained model in the `models/` folder.

## Usage
Once the installation is done you can use the API locally.
First run the service:
```bash
python service/spam_classification_service.py
```
Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
You can then interact with the API by entering the email text in the text area and clicking the `PREDICT!` button to get the result.

## Output format
The API returns a JSON response with the following fields:

```json
{
  "text": "Dear Hiring Manager, Please      find attached my application. Sincerely, Me.",
  "cleaned_text": "Dear Hiring Manager, Please find attached my application. Sincerely, Me.",
  "prediction": "ham",
  "confidence": 0.82,
  "timestamp": "2026-04-23T12:00:00"
}
```
- `text`: original input text
- `cleaned_text`: preprocessed text (extra spaces removed)
- `prediction`: model output (spam or ham)
- `confidence`: model confidence score (between 0 and 1)
- `timestamp`: time of the prediction
