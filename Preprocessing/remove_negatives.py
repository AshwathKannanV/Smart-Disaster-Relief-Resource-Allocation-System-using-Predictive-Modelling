import pandas as pd

def clean_structured_ml_file(filepath="Data/CDD_structured_ml_ready.xlsx"):
    df = pd.read_excel(filepath)

    # Remove rows with any negative numeric values
    numeric_cols = df.select_dtypes(include='number').columns
    df = df[(df[numeric_cols] >= 0).all(axis=1)]

    # Save over the same file
    df.to_excel(filepath, index=False)
    print(f"✅ Cleaned negative values in: {filepath}")
    print(f"📊 Remaining rows: {df.shape[0]}")
    return df


def clean_allocation_result(filepath="Data/resource_allocation_result.xlsx"):
    df = pd.read_excel(filepath)

    # Remove rows with any negative numeric values
    numeric_cols = df.select_dtypes(include='number').columns
    df = df[(df[numeric_cols] >= 0).all(axis=1)]

    # Save over same file
    df.to_excel(filepath, index=False)
    print(f"✅ Cleaned negative values in: {filepath}")
    print(f"📦 Remaining rows: {df.shape[0]}")
    return df
