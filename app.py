import streamlit as st
import numpy as np
import pandas as pd
from utils import *

st.set_page_config(layout='wide')

gender_label_encoder, geo_oh_encoder, scaler, model = load_artifacts()

st.title('Customer Churn Prediction')

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', geo_oh_encoder.categories_[0])
    gender = st.selectbox('Gender', gender_label_encoder.classes_)
    age = st.slider('Age', 18, 92, value=42)
    balance = st.number_input('Enter Balance', min_value=0.0, step=1000.0, value=101348.88)
    credit_score = st.number_input('Enter Credit Score', min_value=0, step=1, value=619)
    estimated_salary = st.number_input('Enter Estimated Salary', min_value=0.0, step=1000.0, value=101348.88)
    tenure = st.slider('Enter Tenure', min_value=0, max_value=10, value=2)
    num_of_products = st.slider('Enter Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
    is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_label_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = geo_oh_encoder.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_oh_encoder.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

with col2:
    st.header(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.header('The customer is likely to churn.')
    else:
        st.header('The customer is not likely to churn.')
