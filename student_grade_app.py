
import streamlit as st
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open("grade_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("📚 Student Grade Predictor")
st.write("Enter student details below to predict the average exam score.")

gender = st.selectbox("Gender", ['female', 'male'])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ['standard', 'free/reduced'])
test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
reading_score = st.slider("Reading Score", 0, 100, 70)
writing_score = st.slider("Writing Score", 0, 100, 70)



input_dict = {
    'gender': gender,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': test_prep,
    'reading score': reading_score,
    'writing score': writing_score
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)


for col in model_columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[model_columns]  


if st.button("Predict Grade"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Average Score: {prediction:.2f} / 100")
