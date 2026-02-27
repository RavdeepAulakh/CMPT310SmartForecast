import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):

    X = df.drop(columns=["student_id", "exam_score"])
    y = df["exam_score"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )