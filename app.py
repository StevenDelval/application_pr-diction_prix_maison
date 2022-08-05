import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('csv/X_train.csv')
X_test = pd.read_csv('csv/X_test.csv')
y_train = pd.read_csv('csv/y_train.csv')
y_test = pd.read_csv('csv/y_test.csv') 
standarisation = StandardScaler()
standarisation.mean_ = 2080.343165401
standarisation.scale_ = 918.1007813027343
standarisation.var_ = 842909.044628691
standarisation.n_samples_seen_ = 1
modelLR = make_pipeline(PolynomialFeatures(2),LinearRegression())
modelLR.fit(np.array(X_train['sqft_living']).reshape(-1, 1),y_train)
score = modelLR.score(np.array(X_test['sqft_living']).reshape(-1, 1),y_test)

st.write(score) 


sqft = st.number_input(label="sqft_living",min_value=0)

if(st.button("Valider")): 
    st.write(int(modelLR.predict(np.array(standarisation.transform(np.array(sqft).reshape(1, -1))).reshape(1,-1))))
