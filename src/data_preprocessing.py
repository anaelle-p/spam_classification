import pandas as pd
import re
from sklearn.model_selection import train_test_split

DATASET_PATH = '../data/spam_email_dataset.csv'

def get_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    df.set_index("email_id", inplace=True)
    df["subject"] = df["subject"].fillna("")
    df["email_text"] = df["email_text"].fillna("")
    df["label_name"] = df["label"].apply(lambda x: "SPAM" if x == 1 else "HAM")
    df["email_full"] = df["subject"] + " - " + df["email_text"]
    df = df[["subject", "email_text", "email_full", "label", "label_name"]]
    return df

def clean(text):
    text = re.sub(" +", " ", text)
    text = text.strip()
    return text

def data_cleaning(df, text_columns=None):
    if text_columns is None:
        text_columns = ["subject", "email_text", "email_full"]
    df = df.copy()
    for col in text_columns:
        df[col] = df[col].apply(lambda x: clean(x))
    return df

def split_data(df, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df["email_full"],
                                                        df["label"],
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=df["label"])
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = data_cleaning(get_data())
    print(df.head())
