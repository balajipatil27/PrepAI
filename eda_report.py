import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_eda_summary(df):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "describe": df.describe(include='all').fillna("N/A").to_html(classes='table', border=0),
        "head": df.head().to_html(classes='table', border=0),
        "tail": df.tail().to_html(classes='table', border=0)
    }

def generate_correlation_plot(df, uid):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return None

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    path = f'static/plots/{uid}_correlation.png'
    plt.savefig(path)
    plt.close()
    return path