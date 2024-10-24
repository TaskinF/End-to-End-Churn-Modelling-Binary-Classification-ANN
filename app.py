import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
ann_model = tf.keras.models.load_model('ann_model.h5')

# Load the encoder and scaler
with open('oh_encoder_geo.pkl','rb') as file:
    oh_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', oh_encoder_geo.encoder_dict_['Geography'])
gender = st.selectbox('Gender', label_encoder_gender.encoder_dict_['Gender'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Preprocessing 
input_df = label_encoder_gender.transform(input_data)
input_df = oh_encoder_geo.transform(input_df)
input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict churn
prediction = ann_model.predict(input_df)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')