import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('minmax_scaler.joblib')  # Load the normalization scaler

# Define columns used in the training process
all_features = [
    'OverallQual', 'GarageCars', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 
    'FullBath', 'YearBuilt'
]

# Define original feature names for display
original_feature_names = {
    'OverallQual': 'Overall Quality',
    'GarageCars': 'Garage Cars',
    'GrLivArea': 'Ground Living Area (sq ft)',
    'GarageArea': 'Garage Area (sq ft)',
    'TotalBsmtSF': 'Total Basement Area (sq ft)',
    'FullBath': 'Full Bathrooms',
    'YearBuilt': 'Year Built'
}

# Title of the app
st.title('🏠 House Price Predictor 🏠')

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    overall_qual = st.sidebar.slider(original_feature_names['OverallQual'], 1, 10, 5)
    garage_cars = st.sidebar.slider(original_feature_names['GarageCars'], 0, 4, 2)
    gr_liv_area = st.sidebar.slider(original_feature_names['GrLivArea'], 334, 5642, 1500)
    garage_area = st.sidebar.slider(original_feature_names['GarageArea'], 0, 1418, 500)
    total_bsmt_sf = st.sidebar.slider(original_feature_names['TotalBsmtSF'], 0, 6110, 800)
    full_bath = st.sidebar.slider(original_feature_names['FullBath'], 0, 4, 2)
    year_built = st.sidebar.slider(original_feature_names['YearBuilt'], 1872, 2010, 1970)
    
    # Create a DataFrame with the input data
    data = {
        'OverallQual': overall_qual,
        'GarageCars': garage_cars,
        'GrLivArea': gr_liv_area,
        'GarageArea': garage_area,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath,
        'YearBuilt': year_built
    }
    
    # Create a DataFrame for user input
    features = pd.DataFrame(data, index=[0])
    
    # Ensure all features are included (fill missing features with default values)
    for col in all_features:
        if col not in features.columns:
            features[col] = 0
    
    # Reorder columns to match the training set
    features = features[all_features]

    return features

input_df = user_input_features()

# Display user input with enhanced formatting
st.subheader('🔍 User Input Parameters')
st.write("Here are the parameters you've provided:")
st.write(input_df.rename(columns=original_feature_names))

# Apply normalization
try:
    # Normalize the features
    normalized_features = scaler.transform(input_df)
except ValueError as e:
    st.error(f"Error during normalization: {e}")
    st.stop()

# Predict house price
try:
    prediction = model.predict(normalized_features)
    predicted_price = prediction.item()  # Convert the prediction to a scalar value
    st.subheader('💰 Predicted House Price')
    st.write(f"${predicted_price:,.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")
