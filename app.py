import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('house_price_model.joblib')

# Title of the app
st.title('House Price Predictor')

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    overall_qual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    gr_liv_area = st.sidebar.slider('Ground Living Area (sq ft)', 334, 5642, 1500)
    garage_cars = st.sidebar.slider('Garage Cars', 0, 4, 2)
    garage_area = st.sidebar.slider('Garage Area (sq ft)', 0, 1418, 500)
    total_bsmt_sf = st.sidebar.slider('Total Basement Area (sq ft)', 0, 6110, 800)
    full_bath = st.sidebar.slider('Full Bathrooms', 0, 4, 2)
    year_built = st.sidebar.slider('Year Built', 1872, 2010, 1970)
    
    data = {
        'OverallQual': overall_qual,
        'GarageCars': garage_cars,
        'GrLivArea': gr_liv_area,
        'GarageArea': garage_area,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath,
        'YearBuilt': year_built
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Predict house price
prediction = model.predict(input_df)

# Extract single prediction value from numpy array and convert to float
predicted_price = prediction.item()  # Use item() method to get the scalar value

# Display prediction
st.subheader('Predicted House Price')
st.write(f"${predicted_price:,.2f}")
