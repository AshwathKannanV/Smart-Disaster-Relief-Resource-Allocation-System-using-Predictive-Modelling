import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

def train_resource_classifier(filepath="Data/CDD_structured_ml_ready.xlsx"):
    df = pd.read_excel(filepath)

    df["RESOURCE TYPES"] = df["RESOURCE TYPES"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df[df["RESOURCE TYPES"].notna()]

    df["MAGNITUDE"] = df["MAGNITUDE"].fillna(0)
    df["ESTIMATED TOTAL COST"] = df["ESTIMATED TOTAL COST"].fillna(0)
    df["UTILITY - PEOPLE AFFECTED"] = df["UTILITY - PEOPLE AFFECTED"].fillna(0)
    df["INJURED / INFECTED"] = df["INJURED / INFECTED"].fillna(0)

    categorical = df[["EVENT CATEGORY", "EVENT GROUP", "EVENT SUBGROUP", "EVENT TYPE"]]
    categorical_encoded = pd.get_dummies(categorical, drop_first=True)

    numerical = df[["MAGNITUDE", "ESTIMATED TOTAL COST", "UTILITY - PEOPLE AFFECTED", "INJURED / INFECTED"]]
    X = pd.concat([numerical, categorical_encoded], axis=1)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["RESOURCE TYPES"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    joblib.dump(model, "Models/resource_classifier.pkl")
    joblib.dump(mlb, "Models/resource_label_binarizer.pkl")
    joblib.dump(X.columns.tolist(), "Models/resource_feature_columns.pkl")

    print("✅ Model, encoder, and feature columns saved.")
    return model, mlb
