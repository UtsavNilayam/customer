import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import pickle
# from keras.models import load_model
from tensorflow.keras.models import load_model
#loading model
# model=tf.keras.load_model('C:\Users\91969\Desktop\Python\ANN classification\model.h5')
model = tf.keras.load_model(r'C:/Users/91969/Desktop/Python/ANN classification/model.h5')

# model = load_model('model.h5')
#load all pickle file
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender =pickle.loads(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    oneot_encoder = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler =pickle.load(file)



input_data = pd.DataFrame({
    'Creditscore' :[credit_score],
    'Age' : [age],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfPrd':[num_of_prd],
    'HasCrCard':[has_cr_card],
    'IsActiveNumber':[is_active_member],
    'EstimatedSalary':[estimared_salary]

})
 #one hot encoding cgeo
geo_encoded=oneot_encoder.transform([[geography]]).tarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=oneot_encoder.get_feature_names_out)

#concat
pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale
scaled_df=scaler.transform(input_data)

#prediction

prediction=model.predict(scaled_df)
prediction_prob=prediction[0][0]

if prediction_prob>0.5:
    st.write("The customer is not likely to churn")

else:
    st.write("The cusstomer likely to churn")
