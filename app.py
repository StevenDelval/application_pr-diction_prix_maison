from math import floor
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('csv/X_train.csv')
X_test = pd.read_csv('csv/X_test.csv')
y_train = pd.read_csv('csv/y_train.csv')
y_test = pd.read_csv('csv/y_test.csv') 


standarisation = StandardScaler()

standarisation.mean_ = np.array([4.05399148e-01, 2.63937766e-03, 2.11584321e+00, 2.08034317e+03,
       5.17073673e+06, 1.50998301e+04, 1.49411928e+00, 1.53732173e-02,
       4.44989813e-02, 2.36154843e-02, 1.47712539e-02, 3.40975181e+00,
       7.65794592e+00, 1.78863151e+03, 3.88436294e+06, 2.91711660e+02,
       2.81046626e+05, 9.61752176e-02, 4.23226523e-02])

standarisation.scale_ = np.array([4.90969122e-01, 5.13070302e-02, 7.68980267e-01, 9.18100781e+02,
       5.40639459e+06, 4.14125906e+04, 5.39672317e-01, 1.23032034e-01,
       2.06200926e-01, 1.51847928e-01, 1.20636081e-01, 6.50455632e-01,
       1.17319112e+00, 8.27744086e+02, 4.14456809e+06, 4.42663454e+02,
       6.38111100e+05, 2.94831384e-01, 2.01324230e-01])

standarisation.n_samples_seen_ = 1
best_alpha=1109.90999999998

modelRid = make_pipeline( PolynomialFeatures(degree=2),Ridge(alpha=best_alpha))
modelRid.fit(X_train,y_train)
score = modelRid.score(X_test,y_test)
titre = "Prédiction prix d'une maison"
original_title = '<p style="font-family:Courier; color:Blue; font-size: 42px;">{} </p>'.format(titre)
st.markdown(original_title, unsafe_allow_html=True)

st.write(score)

caracteristique_maison = [0 for _ in range(19)]

nb_bedrooms = st.number_input(label="bedrooms :",min_value=0,step=1, key="bedrooms :")
if 4<= nb_bedrooms <=6:
  caracteristique_maison[0] = 1
  caracteristique_maison[1] = 0
elif 7<= nb_bedrooms <=9:
  caracteristique_maison[0] = 0
  caracteristique_maison[1] = 1

nb_bathrooms = st.number_input(label="bathrooms :",min_value=0.,step=1.,format="%.2f", key="bathrooms :")
caracteristique_maison[2] = nb_bathrooms

sqft_living = st.number_input(label="sqft_living :",min_value=0, key="sqft_living :")
caracteristique_maison[3]=sqft_living
caracteristique_maison[4]=sqft_living**2


sqft_lot  = st.number_input(label="sqft_lot :",min_value=0, key="sqft_lot :")
caracteristique_maison[5] = sqft_lot

nb_floor = st.number_input(label="floors :",min_value=0.,step=1.,format="%.2f", key="floors :")
caracteristique_maison[6] = nb_floor

view = st.number_input(label="view :",min_value=0,step=1,max_value=4, key="view :")
if view == 1:
  caracteristique_maison[7] = 1
  caracteristique_maison[8] = 0
  caracteristique_maison[9] = 0
  caracteristique_maison[10] = 0
elif view == 2:
  caracteristique_maison[7] = 0
  caracteristique_maison[8] = 1
  caracteristique_maison[9] = 0
  caracteristique_maison[10] = 0
elif view == 3:
  caracteristique_maison[7] = 0
  caracteristique_maison[8] = 0
  caracteristique_maison[9] = 1
  caracteristique_maison[10] = 0
elif view == 4:
  caracteristique_maison[7] = 0
  caracteristique_maison[8] = 0
  caracteristique_maison[9] = 0
  caracteristique_maison[10] = 1

condition = st.number_input(label="condition :",min_value=0,step=1,max_value=4, key="view :")
caracteristique_maison[11] = condition

grade = st.number_input(label="grade :",min_value=1,step=1,max_value=13, key="grade :")
caracteristique_maison[12] = grade

sqft_above = st.number_input(label="sqft_above  :",min_value=0, key="sqft_above :")
caracteristique_maison[13] = sqft_above
caracteristique_maison[14] = sqft_above**2


sqft_basement = st.number_input(label="sqft_basement :",min_value=0, key="sqft_basement :")
caracteristique_maison[15] = sqft_basement
caracteristique_maison[16] = sqft_basement**2

quartier = st.number_input(label="zip code :",min_value=98000,max_value=99000, key="zip code :")
if quartier in [98004,98006,98033,98039,98040,98105,98112]:
  caracteristique_maison[17] = 1
else:
  caracteristique_maison[17] = 0

is_renovated = st.radio(
     "Est elle rénovée",
     ('Oui', 'Non'))
if is_renovated == 'Oui':
  caracteristique_maison[18] = 1
else:
  caracteristique_maison[18] = 0

if(st.button("Valider")):
    
    
    predic = int(modelRid.predict(np.array(standarisation.transform(np.array(caracteristique_maison).reshape(1, -1)))))
    
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 24px;">{}</p>'.format(predic)
    st.markdown(new_title, unsafe_allow_html=True)