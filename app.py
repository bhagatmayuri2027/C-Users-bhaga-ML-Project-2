import streamlit as st
import numpy as np
import pandas as pd
import pickle

model=pickle.load(open("model.pkl","rb"))
sc=pickle.load(open("sc.pkl","rb"))

st.title ("Diabetic Patient Prediction")
col1, col2= st.columns(2)

with col1:
    Pregnancies=st.slider("Pregnancies",0,17,1)
    BloodPressure=st.slider("Blood Pressure",40,140,72)
    Insulin=st.slider("Insulin",15,300,80)
    DiabetesPedigreeFunction =st.number_input("Diabetes Pedigree Function",min_value=0.05,max_value=3.0,value=0.47,step=0.001,format="%.3f")

with col2:
    Glucose=st.slider("Glucose",50,200,120)
    SkinThickness=st.slider("SkinThickness",7,99,20)
    BMI=st.slider("BMI",18.0,50.0,32.0,step=0.1)
    age=st.slider("Age",21,99,30)

if st.button("Predict"):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure',
               'SkinThickness', 'Insulin', 'BMI',
               'DiabetesPedigreeFunction', 'Age']

    myinput = [[Pregnancies, Glucose, BloodPressure,
                SkinThickness, Insulin, BMI,
                DiabetesPedigreeFunction, age]]

    myinput = pd.DataFrame(myinput, columns=columns)

    myinput = pd.DataFrame(sc.transform(myinput), columns=columns)

    result = model.predict(myinput)

    if result[0] == 0:
        st.error("No. Patient is not diabetic")
    else:
        st.success("Yes. Patient is diabetic")