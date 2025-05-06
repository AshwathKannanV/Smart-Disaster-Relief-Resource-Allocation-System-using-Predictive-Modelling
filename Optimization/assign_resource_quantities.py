import pandas as pd
import ast

# Define fallback resource types in case model returns empty list
FALLBACK_RESOURCES = ['food', 'clean water', 'shelter', 'medical kits']

# Define max resource allocation proportions based on severity
RESOURCE_WEIGHTS = {
    'food': 0.2,
    'clean water': 0.2,
    'shelter': 0.15,
    'medical kits': 0.15,
    'power restoration': 0.1,
    'drainage pumps': 0.05,
    'evacuation': 0.1,
    'debris removal': 0.025,
    'tents': 0.025
}

def assign_resource_quantities(path):
    df = pd.read_excel(path)

    if 'RESOURCE TYPES' not in df.columns:
        raise ValueError("RESOURCE TYPES column missing. Run resource_labeler first.")
    
    df.fillna({'RESOURCE TYPES': '[]'}, inplace=True)

    for resource in RESOURCE_WEIGHTS:
        if resource.upper() not in df.columns:
            df[resource.upper()] = 0

    for idx, row in df.iterrows():
        severity = row.get('SEVERITY SCORE', 0)
        total_units = row.get('ALLOCATED UNITS', 0)

        # If severity is 0, don't assign resources
        if severity <= 0 or total_units <= 0:
            continue

        # Parse resource types
        raw_resources = row['RESOURCE TYPES']
        try:
            resource_types = ast.literal_eval(raw_resources) if isinstance(raw_resources, str) else raw_resources
        except Exception:
            resource_types = []

        if not resource_types:
            resource_types = FALLBACK_RESOURCES

        valid_types = [r for r in resource_types if r in RESOURCE_WEIGHTS]

        if not valid_types:
            valid_types = FALLBACK_RESOURCES

        # Normalize resource weights
        total_weight = sum(RESOURCE_WEIGHTS[r] for r in valid_types)
        for r in valid_types:
            allocation = round((RESOURCE_WEIGHTS[r] / total_weight) * total_units)
            df.at[idx, r.upper()] = allocation

    # Save updated Excel file (overwrite for now)
    df.to_excel(path, index=False)
    print(f"[INFO] Resource quantities successfully assigned in: {path}")
