import pandas as pd

def merge_resource_types(source_file="Data/CDD_structured_ml_ready.xlsx",
                         allocation_file="Data/resource_allocation_result.xlsx"):
    df_source = pd.read_excel(source_file)
    df_alloc = pd.read_excel(allocation_file)

    if "RESOURCE TYPES" not in df_source.columns:
        raise ValueError("RESOURCE TYPES column not found in source dataset.")

    # Merge using PLACE (or a better unique key if available)
    df_merged = pd.merge(df_alloc, df_source[["PLACE", "RESOURCE TYPES"]], on="PLACE", how="left")

    # Save updated file
    df_merged.to_excel(allocation_file, index=False)
    print(f"✅ RESOURCE TYPES added to: {allocation_file}")
    return df_merged
