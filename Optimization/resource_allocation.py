import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

def optimize_allocation(data_path, total_supply=10000):
    # Load dataset
    df = pd.read_excel(data_path)

    # Sort zones by severity (optional)
    df = df.sort_values("SEVERITY SCORE", ascending=False).reset_index(drop=True)

    # LP Problem: Maximize severity-weighted resource distribution
    prob = LpProblem("Disaster_Resource_Allocation", LpMaximize)

    # Decision variables: resource units allocated to each disaster zone
    allocations = {i: LpVariable(f"x_{i}", lowBound=0) for i in df.index}

    # Objective: Maximize total severity impact
    prob += lpSum([allocations[i] * df.loc[i, "SEVERITY SCORE"] for i in df.index])

    # Constraint: Total allocation cannot exceed total supply
    prob += lpSum([allocations[i] for i in df.index]) <= total_supply

    # NEW: Constraint - Limit per zone (max allocation = 20% of total supply)
    max_per_zone = total_supply * 0.2
    for i in df.index:
        prob += allocations[i] <= max_per_zone

    # Solve the problem
    prob.solve()

    # Add results to the DataFrame
    df["ALLOCATED UNITS (OPTIMIZED)"] = [allocations[i].varValue for i in df.index]

    # BACKUP STRATEGY: Proportional Allocation (for comparison)
    total_severity = df["SEVERITY SCORE"].sum()
    df["ALLOCATED UNITS (PROPORTIONAL)"] = (df["SEVERITY SCORE"] / total_severity) * total_supply

    # Save to file
    output_path = "data/resource_allocation_result.xlsx"
    df.to_excel(output_path, index=False)

    print(f"✅ Optimization Status: {LpStatus[prob.status]}")
    print(f"🧮 Total Allocated (Optimized): {df['ALLOCATED UNITS (OPTIMIZED)'].sum():.0f} / {total_supply}")
    print(f"📁 Results saved to: {output_path}")

    return df
