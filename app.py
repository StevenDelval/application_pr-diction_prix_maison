"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

df_preparation = pd.read_csv("csv/df_preparation.csv")

modelLR = make_pipeline( PolynomialFeatures(2),LinearRegression())
modelLR.fit(np.array(df_preparation['sqft_living']).reshape(-1, 1),df_preparation['price'])
score = modelLR.score(np.array(df_preparation['sqft_living']).reshape(-1, 1),df_preparation['price'])

st.write(score) 


sqft = st.number_input(label="sqft_living",min_value=0)
  
if(st.button("Valider")): 
    st.write(int(modelLR.predict(np.array(sqft).reshape(1,-1 ))[0]))
