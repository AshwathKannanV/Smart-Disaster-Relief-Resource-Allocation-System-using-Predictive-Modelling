import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the structured file
df = pd.read_excel("CDD_cleaned_structured.xlsx")

# --- Convert Dates ---
df["EVENT START DATE"] = pd.to_datetime(df["EVENT START DATE"], errors="coerce")
df["EVENT END DATE"] = pd.to_datetime(df["EVENT END DATE"], errors="coerce")

# Date features
df["YEAR"] = df["EVENT START DATE"].dt.year
df["MONTH"] = df["EVENT START DATE"].dt.month
df["EVENT DURATION"] = (df["EVENT END DATE"] - df["EVENT START DATE"]).dt.days.fillna(0)

# --- Encode Categorical Columns ---
label_cols = ['EVENT CATEGORY', 'EVENT GROUP', 'EVENT SUBGROUP', 'EVENT TYPE', 'PLACE']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# --- Feature Engineering: Severity Score (with original values) ---
df["SEVERITY SCORE"] = (
    df["FATALITIES"] * 3 +
    df["INJURED / INFECTED"] * 2 +
    df["EVACUATED"] * 1.5 +
    df["UTILITY - PEOPLE AFFECTED"] * 1 +
    df["ESTIMATED TOTAL COST"]
)

# Save it
df.to_csv("preprocessed_disaster_data_raw.csv", index=False)
print("✅ Saved as 'preprocessed_disaster_data_raw.csv' with original numeric values")
