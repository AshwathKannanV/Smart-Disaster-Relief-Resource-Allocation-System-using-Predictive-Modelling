from Preprocessing.clean_data import load_and_clean
from Features.feature_engineering import add_features
from EDA.eda_visuals import run_eda
from Optimization.resource_allocation import optimize_allocation
import pandas as pd
from Preprocessing.remove_negatives import clean_structured_ml_file, clean_allocation_result
from Preprocessing.clean_allocation_output import finalize_allocation_file
from Features.resource_labeler import assign_resource_types
from Optimization.add_resource_types_to_allocation import merge_resource_types
from Optimization.assign_resource_quantities import assign_resource_quantities
from Models.resource_type_classifier import train_resource_classifier
from Models.resource_predictor import predict_resources

# Step 1: Load and clean
df = load_and_clean("data/CDD_cleaned_structured.xlsx")

# Step 2: Feature engineering
df = add_features(df)

# Step 3: Save processed data
df.to_excel("data/CDD_structured_ml_ready.xlsx", index=False)
print("✅ Feature-engineered dataset saved!")

# Step 4: EDA
run_eda(df)

# Step 5: Remove negatives in both working datasets
clean_structured_ml_file("Data/CDD_structured_ml_ready.xlsx")
clean_allocation_result("Data/resource_allocation_result.xlsx")

# Step 6: Clean resource allocation file (remove optimized, round proportional)
finalize_allocation_file("Data/resource_allocation_result.xlsx")

# Step 7: Add resource types based on disaster type
assign_resource_types("Data/CDD_structured_ml_ready.xlsx")

# Step 7.5: Merge RESOURCE TYPES into allocation result
merge_resource_types()

# Step 8: Assign quantities for each resource
assign_resource_quantities("Data/resource_allocation_result.xlsx")

# Step 9: Train multi-label resource prediction model
train_resource_classifier("Data/CDD_structured_ml_ready.xlsx")	














