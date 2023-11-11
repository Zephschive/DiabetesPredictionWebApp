import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

oHe = OneHotEncoder()
ct = ColumnTransformer(transformers=[('encoder', oHe, [3])], remainder='passthrough')

loaded_model = pickle.load(open("C:/Users/zepha/trained_model_diabetes.sav", 'rb'))

def DiabetesPrediction(input_data):
  
    input_data_as_numpy_array = np.array(input_data, dtype=np.float64).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    if prediction[0] == 0:
        return 'You are not a diabetic patient'
    else:
        return 'You are a diabetic patient'

def save_feedback(feedback_data):
    feedback_df = pd.DataFrame(feedback_data, columns=["Timestamp", "Feedback"])
    feedback_df.to_csv("feedback_data.csv", mode="a", header=False, index=False)

def main():
  
    st.title('Intelligent Diabetes Patient Predictor')

   
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness = st.text_input("SkinThickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
    Age = st.text_input("Age")
    feedback = st.text_area("Give Us Your Feedback")


    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    try:
        # Validate the other inputs to ensure they are numeric
        if None not in input_data and all(val.replace(".", "", 1).replace("-", "", 1).isnumeric() for val in input_data):
            # Convert Age to integer
            Age = int(Age)

            Result = ''
            # Creating a button for prediction
            if st.button('Diabetes Prediction Result'):
                Result = DiabetesPrediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

            st.success(Result)
            feedback_data = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), feedback)]
            save_feedback(feedback_data)
        else:
            st.error("Please enter valid numeric values for the input fields, and ensure Age is an integer.")
    except ValueError:
        st.error("Please enter a valid integer for Age.")

if __name__ == '__main__':
    main()
