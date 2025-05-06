import pandas as pd

def add_features(df):
    # Event Duration
    df["EVENT DURATION"] = (df["EVENT END DATE"] - df["EVENT START DATE"]).dt.days.fillna(0)

    # Total Human Impact
    df["TOTAL HUMAN IMPACT"] = (
        df.get("FATALITIES", 0) +
        df.get("INJURED / INFECTED", 0) +
        df.get("EVACUATED", 0) +
        df.get("UTILITY - PEOPLE AFFECTED", 0)
    )

    # Financial Support Sum
    financial_cols = [
        'FEDERAL DFAA PAYMENTS', 'PROVINCIAL DFAA PAYMENTS',
        'PROVINCIAL DEPARTMENT PAYMENTS', 'MUNICIPAL COSTS',
        'OGD COSTS', 'INSURANCE PAYMENTS', 'NGO PAYMENTS'
    ]
    df["TOTAL FINANCIAL SUPPORT"] = df[financial_cols].sum(axis=1)

    # Cost per Person
    df["COST PER PERSON"] = df["ESTIMATED TOTAL COST"] / (df["UTILITY - PEOPLE AFFECTED"] + 1e-5)

    # Severity Score
    df["SEVERITY SCORE"] = (
        df["TOTAL HUMAN IMPACT"] * 0.4 +
        df["ESTIMATED TOTAL COST"] * 0.3 +
        df["EVENT DURATION"] * 0.3
    )

    return df
