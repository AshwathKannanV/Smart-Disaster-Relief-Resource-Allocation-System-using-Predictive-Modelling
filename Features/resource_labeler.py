import pandas as pd
import math

# Step 1: Map disaster event type to needed resources
def assign_resource_types(filepath="Data/CDD_structured_ml_ready.xlsx"):
    df = pd.read_excel(filepath)

    resource_mapping = {
        "Storm": ["shelter", "food", "power restoration"],
        "Flood": ["clean water", "food", "drainage pumps"],
        "Fire": ["medical kits", "evacuation"],
        "Earthquake": ["tents", "medical kits", "debris removal"],
        "Hurricane": ["shelter", "food", "medical kits", "power restoration"],
        "Tornado": ["evacuation", "power restoration", "shelter"],
        "Drought": ["clean water", "medical kits"]
    }

    # Assign resource labels
    df["RESOURCE TYPES"] = df["EVENT TYPE"].map(resource_mapping)

    # Remove rows with no resource mapping
    df = df.dropna(subset=["RESOURCE TYPES"])

    df.to_excel(filepath, index=False)
    print(f"✅ RESOURCE TYPES added to {filepath}")
    return df


# Step 2: Estimate quantity for each resource based on people affected, injuries, etc.
def assign_resource_quantities(filepath="Data/CDD_structured_ml_ready.xlsx"):
    df = pd.read_excel(filepath)

    if "RESOURCE TYPES" not in df.columns:
        raise ValueError("RESOURCE TYPES column not found. Run assign_resource_types first.")

    df["UTILITY - PEOPLE AFFECTED"] = df["UTILITY - PEOPLE AFFECTED"].fillna(0)
    df["INJURED / INFECTED"] = df["INJURED / INFECTED"].fillna(0)

    resource_types = ["food", "clean water", "shelter", "medical kits", "power restoration",
                      "drainage pumps", "evacuation", "debris removal", "tents"]

    for res in resource_types:
        df[res.upper()] = 0  # New columns

    for idx, row in df.iterrows():
        resources = row["RESOURCE TYPES"]
        people = row["UTILITY - PEOPLE AFFECTED"]
        injured = row["INJURED / INFECTED"]

        if not isinstance(resources, list):
            continue

        for res in resources:
            res = res.lower()
            if res == "food":
                df.at[idx, "FOOD"] = math.ceil(people * 2)
            elif res == "clean water":
                df.at[idx, "CLEAN WATER"] = math.ceil(people * 3)
            elif res == "shelter":
                df.at[idx, "SHELTER"] = math.ceil(people / 4)
            elif res == "medical kits":
                df.at[idx, "MEDICAL KITS"] = math.ceil(people * 0.1 + injured / 5)
            elif res == "tents":
                df.at[idx, "TENTS"] = math.ceil(people / 5)
            elif res == "evacuation":
                df.at[idx, "EVACUATION"] = math.ceil(people)
            elif res == "power restoration":
                df.at[idx, "POWER RESTORATION"] = 1
            elif res == "drainage pumps":
                df.at[idx, "DRAINAGE PUMPS"] = 1
            elif res == "debris removal":
                df.at[idx, "DEBRIS REMOVAL"] = 1

    df.to_excel(filepath, index=False)
    print(f"✅ Estimated resource quantities added to {filepath}")
    return df
