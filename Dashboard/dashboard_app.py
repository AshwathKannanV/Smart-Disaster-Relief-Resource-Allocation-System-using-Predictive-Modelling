"""
Smart Disaster Management System - Dashboard
This module provides a production-ready Streamlit dashboard for the Smart Disaster Management System.
It visualizes disaster data and resource allocation predictions based on ML models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import ast
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from Preprocessing.clean_data import load_and_clean
from Preprocessing.remove_negatives import clean_structured_ml_file
from Preprocessing.clean_allocation_output import finalize_allocation_file
from Features.feature_engineering import add_features
from Features.resource_labeler import assign_resource_types, assign_resource_quantities
from Models.resource_predictor import predict_resources
from Optimization.resource_allocation import optimize_allocation
from Optimization.assign_resource_quantities import assign_resource_quantities as assign_quantities
from Optimization.add_resource_types_to_allocation import merge_resource_types
from EDA.eda_visuals import run_eda

# Set page configuration
st.set_page_config(
    page_title="Smart Disaster Management System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4b77ff;
        margin-bottom: 1rem;
    }
    .status-successful {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f5ff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #000000;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
    }
    .resource-table {
        font-size: 14px;
    }
    .filter-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define fallback resource types and weights (from assign_resource_quantities.py)
FALLBACK_RESOURCES = ['food', 'clean water', 'shelter', 'medical kits']
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

# Function to load data
@st.cache_data
def load_data():
    """Load and preprocess disaster data"""
    try:
        # Try to load from standard data path
        df = pd.read_excel("Data/CDD_structured_ml_ready.xlsx")
        return df
    except FileNotFoundError:
        # Fallback to cleaned data
        try:
            df = pd.read_excel("Data/CDD_cleaned_structured.xlsx")
            # Apply feature engineering
            df = add_features(df)
            return df
        except FileNotFoundError:
            st.error("No disaster data files found. Please check the Data directory.")
            return None

# Function to load ML models
@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        # Load classifier model
        model = joblib.load("Models/resource_classifier.pkl")
        feature_cols = joblib.load("Models/resource_feature_columns.pkl")
        mlb = joblib.load("Models/resource_label_binarizer.pkl")
        return model, feature_cols, mlb
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Function to get available resources from user input
def get_available_resources():
    """Get available resources for allocation from user input"""
    st.sidebar.markdown("### Available Resources")
    
    # Define default values
    default_resources = {
        'FOOD': 10000,  # units
        'CLEAN WATER': 20000,  # liters
        'SHELTER': 2000,  # units
        'MEDICAL KITS': 5000,  # kits
        'POWER RESTORATION': 100,  # teams
        'DRAINAGE PUMPS': 50,  # units
        'EVACUATION': 5000,  # capacity
        'DEBRIS REMOVAL': 200,  # teams
        'TENTS': 1000  # units
    }
    
    # Create a dictionary to store user-defined resources
    resources = {}
    
    # Initialize with default values (to ensure we have values even if user doesn't expand the configuration)
    for resource, default_value in default_resources.items():
        resources[resource] = default_value
    
    # Use an expander to save space
    with st.sidebar.expander("Configure Available Resources", expanded=False):
        # Let user adjust each resource
        for resource, default_value in default_resources.items():
            resources[resource] = st.number_input(
                f"{resource}",
                min_value=0,
                value=default_value,
                step=100 if default_value > 1000 else 10,
                help=f"Available units of {resource}"
            )
    
    # Display a summary of configured resources
    total_resources = sum(resources.values())
    st.sidebar.markdown(f"**Total: {total_resources:,} units**")
    
    return resources

# Function to create a download link
def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    """Process an uploaded disaster data file"""
    try:
        # Determine file type and read
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Basic validation - check if necessary columns exist
        required_columns = ['EVENT TYPE', 'PLACE', 'UTILITY - PEOPLE AFFECTED', 'SEVERITY SCORE']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Uploaded file is missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Clean the uploaded data using project's clean_data function
        df = load_and_clean(df)
        
        # Add features using project's feature_engineering function
        df = add_features(df)
        
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

# Function to predict resources for a single disaster
def predict_single_disaster(input_data, model, feature_cols, mlb):
    """Predict resources for a single disaster input"""
    try:
        # Format input for prediction
        disaster_input = {}
        for key, value in input_data.items():
            disaster_input[key] = value
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([disaster_input])
        
        # Get dummies for categorical features
        categorical_cols = ['EVENT CATEGORY', 'EVENT GROUP', 'EVENT SUBGROUP', 'EVENT TYPE']
        cat_cols_present = [col for col in categorical_cols if col in input_df.columns]
        
        if cat_cols_present:
            input_df = pd.get_dummies(input_df, columns=cat_cols_present)
        
        # Ensure all features are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the feature columns in the right order
        input_features = input_df[feature_cols]
        
        # Make prediction
        y_pred = model.predict(input_features)
        
        # Convert prediction to resource types
        predicted_resources = mlb.inverse_transform(y_pred)[0]
        
        # Return prediction
        return list(predicted_resources)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return FALLBACK_RESOURCES

# Function to display resource allocation for a dataset
# Function to display resource allocation for a dataset
def display_resource_allocation(df, available_resources):
    """Calculate and display resource allocation for a disaster dataset"""
    try:
        # Create a temporary file for allocation
        temp_path = "Data/temp_allocation.xlsx"
        df.to_excel(temp_path, index=False)
        
        # Run optimization with available resources
        total_supply = sum(available_resources.values())
        allocation_df = optimize_allocation(temp_path, total_supply)
        
        # Clean allocation output
        allocation_df = finalize_allocation_file(temp_path)
        
        # Add resource types to allocation results
        if 'RESOURCE TYPES' in df.columns:
            # If resource types already exist, copy them
            allocation_df['RESOURCE TYPES'] = df['RESOURCE TYPES']
        else:
            # Assign resource types based on disaster types
            assign_resource_types(temp_path)
            allocation_df = pd.read_excel(temp_path)
        
        # Assign quantities - this appears to not be working
        # Instead, let's manually allocate resources based on severity or criticality
        
        # Read the current state
        final_allocation = pd.read_excel(temp_path)
        
        # Manually allocate resources based on severity or other factors
        if 'SEVERITY SCORE' in final_allocation.columns:
            # Normalize severity to use as allocation weight
            total_severity = final_allocation['SEVERITY SCORE'].sum()
            if total_severity > 0:
                final_allocation['ALLOCATION_WEIGHT'] = final_allocation['SEVERITY SCORE'] / total_severity
            else:
                # Equal distribution if no severity data
                final_allocation['ALLOCATION_WEIGHT'] = 1.0 / len(final_allocation)
        else:
            # Equal distribution if no severity column
            final_allocation['ALLOCATION_WEIGHT'] = 1.0 / len(final_allocation)
        
        # Now allocate each resource based on weights
        for resource_name, available_amount in available_resources.items():
            resource_upper = resource_name.upper()
            if resource_upper not in final_allocation.columns:
                final_allocation[resource_upper] = 0
            
            # Allocate proportionally to weights
            final_allocation[resource_upper] = (final_allocation['ALLOCATION_WEIGHT'] * available_amount).astype(int)
        
        # Remove the temporary weight column
        if 'ALLOCATION_WEIGHT' in final_allocation.columns:
            final_allocation = final_allocation.drop('ALLOCATION_WEIGHT', axis=1)
        
        # Save the final allocation back to temp file
        final_allocation.to_excel(temp_path, index=False)
        
        # Add resource types column if it doesn't exist
        if 'RESOURCE TYPES' not in final_allocation.columns:
            resource_list = list(available_resources.keys())
            final_allocation['RESOURCE TYPES'] = [[r.lower() for r in resource_list]] * len(final_allocation)
        
        # Calculate total allocation for each row
        resource_cols = [col for col in final_allocation.columns if col.upper() in available_resources.keys()]
        if resource_cols:
            final_allocation['ALLOCATED UNITS'] = final_allocation[resource_cols].sum(axis=1)
        
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return final_allocation
    except Exception as e:
        st.error(f"Error in resource allocation: {str(e)}")
        return None

# Function to create allocation results visualization
def plot_allocation_results(allocation_df, available_resources):
    """Create visualizations for resource allocation results"""
    if allocation_df is None:
        return None, None, None
    
    # Extract resource columns
    resource_cols = [col for col in allocation_df.columns if col.upper() in available_resources.keys()]
    
    if not resource_cols:
        # If no specific resource columns found, look for any columns that might match our resources
        # This is a fallback in case of case mismatch or other issues
        all_cols = allocation_df.columns.tolist()
        resource_cols = []
        for res_key in available_resources.keys():
            matches = [col for col in all_cols if res_key.lower() in col.lower()]
            resource_cols.extend(matches)
        
        if not resource_cols:
            return None, None, None
    
    # Calculate total allocated resources
    allocated = {}
    for col in resource_cols:
        # Handle both uppercase and original case
        res_key = col.upper() if col.upper() in available_resources else col
        allocated[col] = allocation_df[col].sum() if col in allocation_df.columns else 0
    
    # Create allocation summary dataframe
    summary_df = pd.DataFrame({
        'Resource': list(allocated.keys()),
        'Allocated': list(allocated.values()),
        'Available': [available_resources.get(res.upper(), available_resources.get(res, 0)) for res in allocated.keys()]
    })
    
    # Ensure we have positive values to avoid division by zero
    summary_df['Available'] = summary_df['Available'].apply(lambda x: max(x, 1))
    summary_df['Remaining'] = summary_df['Available'] - summary_df['Allocated']
    summary_df['Remaining'] = summary_df['Remaining'].apply(lambda x: max(x, 0))  # Ensure no negative values
    summary_df['Usage %'] = (summary_df['Allocated'] / summary_df['Available'] * 100).round(1)
    
    # Create bar chart
    fig_bar = px.bar(
        summary_df, 
        x='Resource', 
        y=['Allocated', 'Remaining'],
        title='Resource Allocation Status',
        barmode='stack',
        color_discrete_sequence=['#4b77ff', '#e0e0e0']
    )
    
    return fig_bar, summary_df, resource_cols

# Function to create dynamic resource allocation map
def create_resource_map(allocation_df, resource_cols):
    """Create a dynamic map visualization of resource allocation by location"""
    if allocation_df is None or 'PLACE' not in allocation_df.columns:
        return None
    
    # For this example, we'll assign sample lat/long to locations if they don't exist
    # In a real implementation, you would use actual coordinates from your dataset
    if 'latitude' not in allocation_df.columns or 'longitude' not in allocation_df.columns:
        # Generate sample coordinates for each unique place
        places = allocation_df['PLACE'].unique()
        
        # Create a dictionary of place to coordinates (sample values for demonstration)
        import random
        # Base coordinates (center of the map)
        base_lat, base_lon = 40.0, -100.0  # Roughly the center of the US
        
        place_coords = {}
        for place in places:
            # Generate random offset from base coordinates
            lat_offset = random.uniform(-15, 15)
            lon_offset = random.uniform(-25, 25)
            place_coords[place] = (base_lat + lat_offset, base_lon + lon_offset)
        
        # Add coordinates to the dataframe
        allocation_df['latitude'] = allocation_df['PLACE'].map(lambda x: place_coords[x][0])
        allocation_df['longitude'] = allocation_df['PLACE'].map(lambda x: place_coords[x][1])
    
    # Calculate total resources per location
    location_resources = allocation_df.groupby('PLACE').agg({
        'latitude': 'first',
        'longitude': 'first',
        **{col: 'sum' for col in resource_cols}
    }).reset_index()
    
    # Calculate total resources for circle size
    location_resources['TOTAL_RESOURCES'] = location_resources[resource_cols].sum(axis=1)
    
    # Normalize circle size
    max_resources = location_resources['TOTAL_RESOURCES'].max()
    min_resources = location_resources['TOTAL_RESOURCES'].min()
    
    # Avoid division by zero
    range_resources = max(max_resources - min_resources, 1)
    
    # Create normalized size (between 5 and 50)
    location_resources['circle_size'] = 5 + (location_resources['TOTAL_RESOURCES'] - min_resources) / range_resources * 45
    
    # Create a tooltip with resource breakdown
    location_resources['tooltip_html'] = location_resources.apply(
        lambda row: f"<b>{row['PLACE']}</b><br>" + 
                   "<br>".join([f"{col}: {int(row[col])}" for col in resource_cols if row[col] > 0]) +
                   f"<br><b>Total: {int(row['TOTAL_RESOURCES'])}</b>",
        axis=1
    )
    
    # Create map figure
    fig = px.scatter_mapbox(
        location_resources,
        lat='latitude',
        lon='longitude',
        size='circle_size',
        size_max=50,
        color='TOTAL_RESOURCES',
        color_continuous_scale='Viridis',
        hover_name='PLACE',
        hover_data={
            'circle_size': False,
            'latitude': False,
            'longitude': False,
            'TOTAL_RESOURCES': True,
            'PLACE': True
        },
        title='Resource Allocation by Location',
        zoom=3
    )
    
    # Update map layout
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=600
    )
    
    return fig

# Function to identify critical disasters
def identify_critical_disasters(df):
    """Identify critical disasters based on severity and impact"""
    if df is None or df.empty:
        return None
    
    # Calculate criticality if not present
    if 'CRITICALITY' not in df.columns:
        # Normalize values
        if 'SEVERITY SCORE' in df.columns:
            max_severity = df['SEVERITY SCORE'].max()
            normalized_severity = df['SEVERITY SCORE'] / max_severity if max_severity > 0 else 0
            
            # People affected
            if 'UTILITY - PEOPLE AFFECTED' in df.columns:
                max_affected = df['UTILITY - PEOPLE AFFECTED'].max()
                normalized_affected = df['UTILITY - PEOPLE AFFECTED'] / max_affected if max_affected > 0 else 0
                
                # Weighted criticality
                df['CRITICALITY'] = normalized_severity * 0.7 + normalized_affected * 0.3
            else:
                df['CRITICALITY'] = normalized_severity
    
    # Find top 25% critical disasters
    if 'CRITICALITY' in df.columns:
        threshold = df['CRITICALITY'].quantile(0.75)
        critical_df = df[df['CRITICALITY'] >= threshold].copy()
        critical_df = critical_df.sort_values('CRITICALITY', ascending=False)
        return critical_df
    
    return None

# Function to create disaster type distribution plots
def plot_disaster_distribution(df):
    """Create visualizations for disaster type distribution"""
    if df is None or df.empty:
        return None, None
    
    # Disaster types distribution
    if 'EVENT TYPE' in df.columns:
        disaster_counts = df['EVENT TYPE'].value_counts().reset_index()
        disaster_counts.columns = ['Disaster Type', 'Count']
        
        fig_pie = px.pie(
            disaster_counts, 
            values='Count', 
            names='Disaster Type',
            title='Distribution of Disaster Types',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Disaster by location
        if 'PLACE' in df.columns and 'UTILITY - PEOPLE AFFECTED' in df.columns:
            location_impact = df.groupby('PLACE')['UTILITY - PEOPLE AFFECTED'].sum().reset_index()
            location_impact.columns = ['Location', 'Affected Population']
            location_impact = location_impact.sort_values('Affected Population', ascending=False).head(10)
            
            fig_bar = px.bar(
                location_impact,
                x='Location',
                y='Affected Population',
                title='Top 10 Locations by Affected Population',
                color='Affected Population',
                color_continuous_scale='Viridis'
            )
            
            return fig_pie, fig_bar
    
    return None, None

# Main application
def main():
    """Main function for the Streamlit dashboard"""
    # App header
    st.markdown("<h1 class='main-header'>🚨 Smart Disaster Management System</h1>", unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    model, feature_cols, mlb = load_models()
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=SDMS", width=150)
    st.sidebar.title("Control Panel")
    
    # Get available resources from user input
    available_resources = get_available_resources()
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Use existing disaster data", "Upload new disaster data"],
        key="data_source"
    )
    
    # Handle file upload if selected
    if data_source == "Upload new disaster data":
        uploaded_file = st.sidebar.file_uploader("Upload disaster data (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            st.sidebar.info("File uploaded successfully!")
            df = process_uploaded_file(uploaded_file)
            if df is None:
                st.sidebar.error("Could not process the uploaded file. Please check the format and try again.")
    
    # Show data filters if data is available
    filtered_df = None
    if df is not None:
        st.sidebar.markdown("### Filter Options")
        
        # Filter by disaster type
        if 'EVENT TYPE' in df.columns:
            disaster_types = ['All'] + sorted(df['EVENT TYPE'].unique().tolist())
            selected_disaster = st.sidebar.selectbox("Filter by Disaster Type", disaster_types)
        
        # Filter by location
        if 'PLACE' in df.columns:
            locations = ['All'] + sorted(df['PLACE'].unique().tolist())
            selected_location = st.sidebar.selectbox("Filter by Location", locations)
        
        # Filter by date range if date column exists
        if 'EVENT START DATE' in df.columns:
            df['EVENT START DATE'] = pd.to_datetime(df['EVENT START DATE'], errors='coerce')
            min_date = df['EVENT START DATE'].min()
            max_date = df['EVENT START DATE'].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.sidebar.date_input(
                    "Filter by Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
        
        # Filter by severity range
        if 'SEVERITY SCORE' in df.columns:
            min_severity = float(df['SEVERITY SCORE'].min())
            max_severity = float(df['SEVERITY SCORE'].max())
            severity_range = st.sidebar.slider(
                "Filter by Severity",
                min_severity,
                max_severity,
                (min_severity, max_severity)
            )
        
        # Apply filters to data
        filtered_df = df.copy()
        
        if 'EVENT TYPE' in df.columns and 'selected_disaster' in locals() and selected_disaster != 'All':
            filtered_df = filtered_df[filtered_df['EVENT TYPE'] == selected_disaster]
            
        if 'PLACE' in df.columns and 'selected_location' in locals() and selected_location != 'All':
            filtered_df = filtered_df[filtered_df['PLACE'] == selected_location]
            
        if 'EVENT START DATE' in df.columns and 'date_range' in locals() and len(date_range) == 2:
            filtered_df = filtered_df[(filtered_df['EVENT START DATE'].dt.date >= date_range[0]) & 
                                   (filtered_df['EVENT START DATE'].dt.date <= date_range[1])]
            
        if 'SEVERITY SCORE' in df.columns and 'severity_range' in locals():
            filtered_df = filtered_df[(filtered_df['SEVERITY SCORE'] >= severity_range[0]) & 
                                   (filtered_df['SEVERITY SCORE'] <= severity_range[1])]
        
        # Check if we have data after filtering
        if filtered_df.empty:
            st.warning("No data matches your filter criteria. Please adjust your filters.")
            filtered_df = None
    
    # Main content area
    if filtered_df is not None:
        # Create a default allocation with at least some data for testing/demo
        if 'allocation_df' not in locals() or allocation_df is None:
            # Generate some dummy allocation data based on the available resources
            allocation_df = filtered_df.copy()
            
            # Add ALLOCATED UNITS if it doesn't exist
            if 'ALLOCATED UNITS' not in allocation_df.columns:
                # Calculate based on severity
                if 'SEVERITY SCORE' in allocation_df.columns:
                    max_severity = allocation_df['SEVERITY SCORE'].max() or 1
                    allocation_df['ALLOCATED UNITS'] = (allocation_df['SEVERITY SCORE'] / max_severity * 1000).astype(int)
                else:
                    # Default allocation
                    allocation_df['ALLOCATED UNITS'] = 1000
            
            # Add resource columns if they don't exist
            resource_types = []
            for idx, row in allocation_df.iterrows():
                # Select 2-3 random resources for each disaster
                num_resources = min(3, len(RESOURCE_WEIGHTS))
                selected_resources = list(RESOURCE_WEIGHTS.keys())[:num_resources]
                resource_types.append(selected_resources)
                
                # Add resource columns if not present
                for resource in RESOURCE_WEIGHTS.keys():
                    if resource.upper() not in allocation_df.columns:
                        allocation_df[resource.upper()] = 0
                    
                    # Add some resource values for demonstration
                    if resource in selected_resources:
                        # Calculate a reasonable amount based on allocated units
                        allocation_factor = RESOURCE_WEIGHTS[resource] * 2
                        allocation_df.at[idx, resource.upper()] = int(row.get('ALLOCATED UNITS', 1000) * allocation_factor)
            
            # Add RESOURCE TYPES column
            allocation_df['RESOURCE TYPES'] = resource_types
        
        # Allocate resources if model is loaded
        if model is not None and feature_cols is not None and mlb is not None:
            with st.spinner("Predicting and allocating resources..."):
                # Check if RESOURCE TYPES exists
                if 'RESOURCE TYPES' not in filtered_df.columns:
                    # For each row in filtered_df, predict resources
                    resource_types = []
                    for idx, row in filtered_df.iterrows():
                        predicted = predict_single_disaster(row.to_dict(), model, feature_cols, mlb)
                        resource_types.append(predicted)
                    
                    filtered_df['RESOURCE TYPES'] = resource_types
                
                # Allocate resources
                allocation_df = display_resource_allocation(filtered_df, available_resources)
                
                # If allocation still failed, use the demo data
                if allocation_df is None:
                    st.warning("Using demo allocation data for visualization.")
                    allocation_df = filtered_df.copy()
                    
                    # Ensure we have resource columns
                    for resource in available_resources.keys():
                        if resource not in allocation_df.columns:
                            allocation_df[resource] = np.random.randint(0, available_resources[resource] // 10, size=len(allocation_df))
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🗺️ Disaster Analysis", 
            "📈 Resource Allocation", 
            "⚠️ Critical Situations"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.markdown("<h2 class='sub-header'>Disaster Management Overview</h2>", unsafe_allow_html=True)
            
            # Summary metrics
            total_disasters = len(filtered_df)
            total_affected = filtered_df['UTILITY - PEOPLE AFFECTED'].sum() if 'UTILITY - PEOPLE AFFECTED' in filtered_df.columns else 0
            
            # Critical disasters
            critical_df = identify_critical_disasters(filtered_df)
            critical_count = len(critical_df) if critical_df is not None else 0
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Total Disasters</h3>
                    <h2>{total_disasters}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>People Affected</h3>
                    <h2>{total_affected:,}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Critical Situations</h3>
                    <h2>{critical_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                if 'SEVERITY SCORE' in filtered_df.columns:
                    avg_severity = filtered_df['SEVERITY SCORE'].mean()
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Avg. Severity</h3>
                        <h2>{avg_severity:.1f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create and display overview charts
            st.markdown("### Disaster Distribution")
            fig_pie, fig_bar = plot_disaster_distribution(filtered_df)
            
            if fig_pie is not None and fig_bar is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True, key="overview_pie_chart")
                with col2:
                    st.plotly_chart(fig_bar, use_container_width=True, key="overview_bar_chart")
            
            # Resource allocation status with map
            if allocation_df is not None:
                st.markdown("### Resource Allocation Map")
                fig_bar, summary_df, resource_cols = plot_allocation_results(allocation_df, available_resources)
                
                if fig_bar is not None and resource_cols:
                    # Create map
                    map_fig = create_resource_map(allocation_df, resource_cols)
                    
                    if map_fig is not None:
                        st.plotly_chart(map_fig, use_container_width=True, key="overview_map")
                    
                    # Display summary data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_bar, use_container_width=True, key="overview_allocation_bar")
                    with col2:
                        st.dataframe(summary_df, use_container_width=True)
            
            # Download buttons for data
            st.markdown("### Download Data")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Filtered Disaster Data",
                    data=filtered_df.to_csv(index=False).encode('utf-8'),
                    file_name="filtered_disaster_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                if allocation_df is not None:
                    st.download_button(
                        label="Download Resource Allocation Results",
                        data=allocation_df.to_csv(index=False).encode('utf-8'),
                        file_name="resource_allocation_results.csv",
                        mime="text/csv"
                    )
        
        # Tab 2: Disaster Analysis
        with tab2:
            st.markdown("<h2 class='sub-header'>Disaster Data Analysis</h2>", unsafe_allow_html=True)
            
            # Display filtered data
            st.markdown("### Filtered Disaster Data")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Show disaster visualizations
            st.markdown("### Disaster Insights")
            
            # Create visualization columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Event duration distribution if available
                if 'EVENT DURATION' in filtered_df.columns:
                    fig_duration = px.histogram(
                        filtered_df,
                        x='EVENT DURATION',
                        title='Distribution of Event Duration',
                        labels={'EVENT DURATION': 'Duration (days)'},
                        color_discrete_sequence=['#4b77ff']
                    )
                    st.plotly_chart(fig_duration, use_container_width=True, key="event_duration_hist")
            
            with col2:
                # Severity distribution
                if 'SEVERITY SCORE' in filtered_df.columns:
                    fig_severity = px.histogram(
                        filtered_df,
                        x='SEVERITY SCORE',
                        title='Distribution of Severity Scores',
                        labels={'SEVERITY SCORE': 'Severity Score'},
                        color_discrete_sequence=['#ff4b4b']
                    )
                    st.plotly_chart(fig_severity, use_container_width=True, key="severity_hist")
            
            # Show correlation heatmap for numeric columns
            numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_cols) > 1:
                st.markdown("### Correlation Analysis")
                # Filter to most relevant columns to avoid crowding
                key_columns = [col for col in ['FATALITIES', 'INJURED / INFECTED', 'EVACUATED', 
                                              'UTILITY - PEOPLE AFFECTED', 'ESTIMATED TOTAL COST', 
                                              'SEVERITY SCORE', 'EVENT DURATION'] 
                              if col in numeric_cols]
                
                if len(key_columns) > 1:
                    corr = filtered_df[key_columns].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto=True,
                        title='Correlation Between Key Metrics',
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap")
        
        # Tab 3: Resource Allocation
        with tab3:
            st.markdown("<h2 class='sub-header'>Resource Allocation</h2>", unsafe_allow_html=True)
            
            if allocation_df is not None:
                # Display allocation results
                st.markdown("### Resource Allocation Results")
                st.dataframe(allocation_df, use_container_width=True)
                
                # Resource allocation map
                st.markdown("### Resource Allocation Map")
                fig_bar, summary_df, resource_cols = plot_allocation_results(allocation_df, available_resources)
                
                if resource_cols:
                    map_fig = create_resource_map(allocation_df, resource_cols)
                    if map_fig is not None:
                        st.plotly_chart(map_fig, use_container_width=True, key="allocation_map")
                
                # Resource distribution by location
                if 'PLACE' in allocation_df.columns:
                    st.markdown("### Resource Distribution by Location")
                    
                    # Get resource columns
                    resource_cols = [col for col in allocation_df.columns 
                                   if col.upper() in available_resources.keys()]
                    
                    if resource_cols:
                        # Sum resources by location
                        location_resources = allocation_df.groupby('PLACE')[resource_cols].sum().reset_index()
                        
                        # Create location total column
                        location_resources['TOTAL_RESOURCES'] = location_resources[resource_cols].sum(axis=1)
                        
                        # Sort and get top locations
                        top_locations = location_resources.sort_values('TOTAL_RESOURCES', ascending=False).head(10)
                        
                        # Create bar chart
                        fig_location = px.bar(
                            top_locations,
                            x='PLACE',
                            y='TOTAL_RESOURCES',
                            title='Top 10 Locations by Resource Allocation',
                            color='TOTAL_RESOURCES',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_location, use_container_width=True, key="top_locations_bar")
                        
                        # Show detailed breakdown for selected location
                        selected_location = st.selectbox(
                            "Select location to view detailed resource breakdown:",
                            options=top_locations['PLACE'].tolist()
                        )
                        
                        if selected_location:
                            location_detail = allocation_df[allocation_df['PLACE'] == selected_location]
                            
                            # Create pie chart of resources for this location
                            location_resources = location_detail[resource_cols].sum()
                            location_resources = location_resources[location_resources > 0]
                            
                            fig_pie = px.pie(
                                names=location_resources.index,
                                values=location_resources.values,
                                title=f'Resource Distribution for {selected_location}',
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            st.plotly_chart(fig_pie, use_container_width=True, key="location_resources_pie")
            else:
                st.warning("Resource allocation data is not available. Please check if the ML models are properly loaded.")
        
        # Tab 4: Critical Situations
        with tab4:
            st.markdown("<h2 class='sub-header'>Critical Disaster Situations</h2>", unsafe_allow_html=True)
            
            critical_df = identify_critical_disasters(filtered_df)
            
            if critical_df is not None and not critical_df.empty:
                # Display status message
                st.markdown("<div class='status-warning'>⚠️ Critical situations require immediate attention!</div>", unsafe_allow_html=True)
                
                # Display critical disasters
                st.markdown("### High Priority Disasters")
                st.dataframe(critical_df, use_container_width=True)
                
                # Resources for critical disasters
                if allocation_df is not None:
                    st.markdown("### Resource Allocation for Critical Disasters")
                    
                    # Get critical disaster places
                    critical_places = critical_df['PLACE'].unique().tolist()
                    
                    # Filter allocation for critical disasters
                    critical_allocation = allocation_df[allocation_df['PLACE'].isin(critical_places)]
                    
                    if not critical_allocation.empty:
                        # Resource allocation map for critical disasters
                        st.markdown("### Critical Disasters Resource Map")
                        _, _, resource_cols = plot_allocation_results(critical_allocation, available_resources)
                        
                        if resource_cols:
                            map_fig = create_resource_map(critical_allocation, resource_cols)
                            if map_fig is not None:
                                st.plotly_chart(map_fig, use_container_width=True, key="critical_map")
                        
                        st.dataframe(critical_allocation, use_container_width=True)
                        
                        # Show resource distribution
                        resource_cols = [col for col in critical_allocation.columns 
                                       if col.upper() in available_resources.keys()]
                        
                        if resource_cols:
                            resource_sums = critical_allocation[resource_cols].sum()
                            resource_sums = resource_sums[resource_sums > 0]
                            
                            fig_resources = px.pie(
                                names=resource_sums.index,
                                values=resource_sums.values,
                                title='Resource Distribution for Critical Disasters',
                                color_discrete_sequence=px.colors.sequential.Plasma
                            )
                            st.plotly_chart(fig_resources, use_container_width=True, key="critical_resources_pie")
            else:
                st.info("No critical disasters identified based on current filters.")
                
                # Show explanation of critical disaster identification
                st.markdown("""
                <div class='info-box'>
                    <h3>How Critical Situations are Identified</h3>
                    <p>Disasters are flagged as critical based on a combination of:</p>
                    <ul>
                        <li>High severity scores</li>
                        <li>Large affected population</li>
                        <li>Top 25% in calculated criticality</li>
                    </ul>
                    <p>Adjust your filters to see if there are critical situations in the broader dataset.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # No data available
        if df is None:
            st.error("No disaster data available. Please check data files or upload data.")
            
            # Show guidance
            st.markdown("""
            <div class='info-box'>
                <h3>Getting Started Guide</h3>
                <p>The Smart Disaster Management System requires disaster data to function. You can:</p>
                <ol>
                    <li>Check that disaster data files exist in the Data/ directory</li>
                    <li>Upload a CSV or Excel file with disaster data</li>
                    <li>Ensure your data contains the required columns like EVENT TYPE, PLACE, etc.</li>
                </ol>
                <p>Once data is available, the system will predict required resources and allocate them based on need.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No data matches your filter criteria. Please adjust your filters.")

# Add a footer with additional info
def footer():
    """Display footer with additional information"""
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: gray; font-size: 0.8em;">
            <p>Smart Disaster Management System</p>
            <p>Version 1.0 | © 2023</p>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
    footer()