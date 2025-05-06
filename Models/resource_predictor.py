import pandas as pd
import joblib

def predict_resources(new_disaster_input: dict):
    model = joblib.load("Models/resource_classifier.pkl")
    mlb = joblib.load("Models/resource_label_binarizer.pkl")
    feature_columns = joblib.load("Models/resource_feature_columns.pkl")

    input_df = pd.DataFrame([new_disaster_input])
    input_df = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    y_pred = model.predict(input_df)
    predicted_resources = mlb.inverse_transform(y_pred)

    return predicted_resources[0]
