import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle

cars_df = pd.read_csv('./cars24_dataset.csv')

st.write(
    """
    # Cars24 Used Car Price Prediction
    """
)

st.dataframe(cars_df.head())

fuel_type = st.selectbox("select the fuel type",['Diesel','Petrol','CNG','LPG','Electric'])
engine = st.slider("set the engine power",500,5000,step=100)
transmission_type = st.selectbox('select the transmission type',['Manual','Automatic'])
seats = st.selectbox('Enter the number of seats',[4,5,7,9,11])

# input_features = [[2018.0,1,4000,fuel_type,transmission_type,19.70,engine,86.30,seats]]

encode_dict = {
    'fuel_type':{'Diesel':0,'Petrol':1,'CNG':2,'LPG':3,'Electric':4},
    'seller_type':{'Dealer':1,'Individual':2,'Trustmark Dealer':3},
    'transmission_type':{'Manual':1,'Automatic':2}  
}
### do encoding, standardization - whatever you did while building model(do it here too)
def model_pred(fuel_type,transmission_type,engine,seats):
    
    ## loading the model
    with open('car_pred','rb') as file:
        reg_model=pickle.load(file)
        input_features = [[2018.0,1,4000,fuel_type,transmission_type,19.70,engine,86.30,seats]]
        return reg_model.predict(input_features)


if (st.button('Predict Price')):
    fuel_type = encode_dict['fuel_type'][fuel_type]
    transmission_type = encode_dict['transmission_type'][transmission_type]

    price = model_pred(fuel_type,transmission_type,engine,seats)

    st.text(f'The Price of the car is {price[0].round(2)} lakh rupees')


