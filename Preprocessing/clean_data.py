import pandas as pd

def load_and_clean(filepath):
    df = pd.read_excel(filepath)

    # Convert dates
    df["EVENT START DATE"] = pd.to_datetime(df["EVENT START DATE"], errors='coerce')
    df["EVENT END DATE"] = pd.to_datetime(df["EVENT END DATE"], errors='coerce')

    # Fill missing values
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    text_cols = ["EVENT CATEGORY", "EVENT GROUP", "EVENT SUBGROUP", "EVENT TYPE", "PLACE", "COMMENTS"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df
