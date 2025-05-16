import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv("simpledata.csv")

types_available = df['type'].unique().tolist()  
subtypes_available = df['subtype'].unique().tolist() 
localities_available = df['province'].unique().tolist()
builCond_available = df['buildingCondition'].unique().tolist()
zipcode_available = df['postCode'].unique().tolist()

with open("gbr_modele.pkl", "rb") as f:
    pipeline = pickle.load(f)
    print(type(pipeline))


st.title("Prediction with GBRmodel")

st.header("Enter criteria :")
type = st.selectbox("Type", types_available) 
subtype = st.selectbox("Subtype",subtypes_available)
bedroomCount = st.number_input("bedroom Count", min_value=0, value=2)
locality = st.selectbox("province", localities_available)
postCode = st.selectbox("ZipCode", zipcode_available)
habitableSurface = st.number_input("Habitable Surface", min_value=10.0, value=60.0)
buildingCondition = st.selectbox("Buiding condition",builCond_available)
facadeCount = st.number_input("Facade count", min_value=1, max_value=6, value=2)


if st.button("Estimation"):
    input_data = np.array([[type, subtype, bedroomCount,locality,postCode,habitableSurface,buildingCondition,facadeCount]])
    prediction = pipeline.predict(input_data)
    st.success(f"Estimated : {prediction[0]}")
