import pandas as pd

def finalize_allocation_file(filepath="Data/resource_allocation_result.xlsx"):
    # Load file
    df = pd.read_excel(filepath)

    # Remove unrealistic optimized allocation
    if "ALLOCATED UNITS (OPTIMIZED)" in df.columns:
        df.drop(columns=["ALLOCATED UNITS (OPTIMIZED)"], inplace=True)

    # Round proportional allocation and rename
    if "ALLOCATED UNITS (PROPORTIONAL)" in df.columns:
        df["ALLOCATED UNITS"] = df["ALLOCATED UNITS (PROPORTIONAL)"].round().astype(int)
        df.drop(columns=["ALLOCATED UNITS (PROPORTIONAL)"], inplace=True)

    # Save it back to the same file
    df.to_excel(filepath, index=False)
    print(f"✅ Allocation file cleaned and updated at: {filepath}")
    return df
