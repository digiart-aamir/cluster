import streamlit as st
import pickle
import pandas as pd
from datetime import datetime

# Load the model
with open('kmeans_model.pkl', 'rb') as file:
    model = pickle.load(file)

# laod scalar tranform
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set custom title and icon
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
)


# Title for the Streamlit app
st.title('Customer Segmentation')


# Create a form for user input
with st.form("customer_form"):
    ID = st.number_input('ID', value=12345)
    Year_Birth = st.number_input('Year of Birth', min_value=1900, max_value=2024, value=1985)
    Education = st.selectbox('Education Level', ['PhD', 'Master', 'Bachelor', 'High School', 'Other'])
    Marital_Status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widow'])
    Income = st.number_input('Income', value=55000)
    Kidhome = st.number_input('Number of kids at home', min_value=0, max_value=10, value=1)
    Teenhome = st.number_input('Number of teenagers at home', min_value=0, max_value=10, value=2)
    Dt_Customer = st.date_input('Date of Enrollment', value=datetime(2023, 5, 15))
    Recency = st.number_input('Recency (days since last purchase)', min_value=0, value=10)
    MntWines = st.number_input('Amount spent on wines', min_value=0, value=250)
    MntFruits = st.number_input('Amount spent on fruits', min_value=0, value=30)
    MntMeatProducts = st.number_input('Amount spent on meat products', min_value=0, value=150)
    MntFishProducts = st.number_input('Amount spent on fish products', min_value=0, value=70)
    MntSweetProducts = st.number_input('Amount spent on sweet products', min_value=0, value=40)
    MntGoldProds = st.number_input('Amount spent on gold products', min_value=0, value=100)
    NumDealsPurchases = st.number_input('Number of deals purchases', min_value=0, value=5)
    NumWebPurchases = st.number_input('Number of web purchases', min_value=0, value=8)
    NumCatalogPurchases = st.number_input('Number of catalog purchases', min_value=0, value=3)
    NumStorePurchases = st.number_input('Number of store purchases', min_value=0, value=12)
    NumWebVisitsMonth = st.number_input('Number of web visits per month', min_value=0, value=15)
    AcceptedCmp3 = st.selectbox('Accepted Campaign 3', [0, 1])
    AcceptedCmp4 = st.selectbox('Accepted Campaign 4', [0, 1])
    AcceptedCmp5 = st.selectbox('Accepted Campaign 5', [0, 1])
    AcceptedCmp1 = st.selectbox('Accepted Campaign 1', [0, 1])
    AcceptedCmp2 = st.selectbox('Accepted Campaign 2', [0, 1])
    Complain = st.selectbox('Complain', [0, 1])
    Z_CostContact = st.number_input('Z Cost Contact', min_value=0, value=3)
    Z_Revenue = st.number_input('Z Revenue', min_value=0.0, value=11.5)
    Response = st.selectbox('Response', [0, 1])

    # Submit button
    submitted = st.form_submit_button("Submit")

# If the form is submitted, create the data dictionary and display it
if submitted:
    # Convert date to the proper format
    Dt_Customer_str = Dt_Customer.strftime('%m-%d-%Y')

    # Create dictionary
    data_dict = {
        'ID': [ID],
        'Year_Birth': [Year_Birth],
        'Education': [Education],
        'Marital_Status': [Marital_Status],
        'Income': [Income],
        'Kidhome': [Kidhome],
        'Teenhome': [Teenhome],
        'Dt_Customer': [Dt_Customer_str],
        'Recency': [Recency],
        'MntWines': [MntWines],
        'MntFruits': [MntFruits],
        'MntMeatProducts': [MntMeatProducts],
        'MntFishProducts': [MntFishProducts],
        'MntSweetProducts': [MntSweetProducts],
        'MntGoldProds': [MntGoldProds],
        'NumDealsPurchases': [NumDealsPurchases],
        'NumWebPurchases': [NumWebPurchases],
        'NumCatalogPurchases': [NumCatalogPurchases],
        'NumStorePurchases': [NumStorePurchases],
        'NumWebVisitsMonth': [NumWebVisitsMonth],
        'AcceptedCmp3': [AcceptedCmp3],
        'AcceptedCmp4': [AcceptedCmp4],
        'AcceptedCmp5': [AcceptedCmp5],
        'AcceptedCmp1': [AcceptedCmp1],
        'AcceptedCmp2': [AcceptedCmp2],
        'Complain': [Complain],
        'Z_CostContact': [Z_CostContact],  # Assuming a constant value
        'Z_Revenue': [Z_Revenue],   # Assuming a constant value
        'Response': [Response]
    }

    # Convert to DataFrame (optional, for visual display or further processing)
    data = pd.DataFrame(data_dict)

    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%m-%d-%Y')
    data['day'] = data['Dt_Customer'].dt.day
    data['month'] = data['Dt_Customer'].dt.month
    data['year'] = data['Dt_Customer'].dt.year

    data = data.drop('Dt_Customer', axis=1)

    # Provided mappings
    education_mapping = {'2n Cycle': 0, 'Basic': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
    marital_status_mapping = {'Absurd': 0, 'Alone': 1, 'Divorced': 2, 'Married': 3, 'Single': 4, 'Together': 5,
                              'Widow': 6, 'YOLO': 7}

    # Encode 'Education' and 'Marital_Status' using the provided mappings
    data['Education'] = data['Education'].map(education_mapping)
    data['Marital_Status'] = data['Marital_Status'].map(marital_status_mapping)

    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)

    st.success(f"The new data belongs to cluster {pred[0]}.")