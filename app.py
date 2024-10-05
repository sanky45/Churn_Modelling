import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
import streamlit as st


model = tf.keras.models.load_model('model.h5')


#LOAD ENCODER AND SCALER AND OHE
with open('label_enco_gender.pkl','rb') as file:
    label_enco_gender = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)




##streamlit app

st.title('Customer Churn Prediction')

#User input

geography = st.selectbox('geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_enco_gender.classes_)
age = st.selectbox('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products= st.slider("Num of products",1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
Is_active_member= st.selectbox('Is acctive member',[0,1])



#Prepare the input data



input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_enco_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [Is_active_member],
'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaled = scaler.transform(input_data_df)

prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("the customer is likely to churn")
else:
    st.write("the customer is not likely to churn")