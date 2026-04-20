import pandas as pd
import re
from sklearn.model_selection import train_test_split

DATASET_PATH = '../data/enron_spam_data.csv'
TEXT_COLUMNS = ["Subject", "Message", "Message_full"]

def get_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    df.set_index("Message ID", inplace=True)
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["Message_full"] = df["Subject"] + " - " + df["Message"]
    df["Label"] = df["Spam/Ham"]
    df.drop_duplicates(subset=["Message_full"], inplace=True)
    df = df[["Subject", "Message", "Message_full", "Label"]]
    return df

def clean(text):
    text = re.sub(" +", " ", text)
    text = text.strip()
    return text

def data_cleaning(df, text_columns=None):
    if text_columns is None:
        text_columns = TEXT_COLUMNS
    df = df.copy()
    for col in text_columns:
        df[col] = df[col].apply(lambda x: clean(x))
    return df

def split_data(df, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df["Message_full"],
                                                        df["Label"],
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=df["Label"])
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = data_cleaning(get_data())
    print(df.head())
