import streamlit as st 
import pandas as pd 
import joblib


model = joblib.load('logistic_heart.pkl')
scler = joblib.load('scler.pkl')
expacted_cloumns = joblib.load ("columns.pkl")

st.title("Heart stroke predication by MMN❤️")
st.markdown('Provide the following details')

age = st.slider('Age',18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain  = st.selectbox("Chest Pain Type",["ATA", "NAP","TA","ASY"])
resting_bp = st.number_input("Restig Blood Pressur (mm Hg)",80,200,120)
cholesterole = st.number_input("Cholesterol (mg/dl)",100,600,200)
fasting_bs = st.selectbox ("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG ", ["Normal","ST","LVH"])
max_hr = st.slider("Max Heart Rate ", 60,220,150)
exercise_agina = st.selectbox("Exercise-Induced Agina",['Y','N'])
oldpeak  = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slop",['Up','Flat','Down'])

if st.button ("Predict"): 
    raw_input = {
        'Age'        : age,
        'RestingBP'  : resting_bp,
        'Cholesterol': cholesterole,
        'FastingBs'  : fasting_bs,
        'MaxHR'      : max_hr,
        'Oldpeak'    : oldpeak,
        'Sex_'+ sex  : 1,
        'ChestPainType_' + chest_pain  : 1,
        'RestingECG_'+ resting_ecg     : 1,
        'ExerciseAngina_'+ exercise_agina: 1,
        'ST_Slope_'+ st_slope         : 1

    }

    input_df = pd.DataFrame([raw_input])
    for col in expacted_cloumns:
        if col not in input_df.columns :
            input_df[col] = 0

    input_df = input_df[expacted_cloumns]

    scaled_input  = scler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]


    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease ")
    else :
        st.success('✅ Low Risk of Heart Disease') 
