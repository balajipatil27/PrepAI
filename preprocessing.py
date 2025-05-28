import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna("Unknown", inplace=True)
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)