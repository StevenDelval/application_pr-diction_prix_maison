import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,learning_curve

X_train = pd.read_csv('csv/X_train.csv')
X_test = pd.read_csv('csv/X_test.csv')
y_train = pd.read_csv('csv/y_train.csv')
y_test = pd.read_csv('csv/y_test.csv') 

numerical_features = [
       'bedrooms', 'bathrooms',
       'sqft_living', 'sqft_living_carre', 'sqft_lot','sqft_above',
       'sqft_basement','sqft_living15', 'sqft_lot15'
       ]
categorical_features = [
       'floors','waterfront','view', 'condition', 'grade','zipcode'
       ]

my_col_trans = ColumnTransformer([
    ("sclal",StandardScaler(),numerical_features),
    ("pol",PolynomialFeatures(),numerical_features),
    ("one_hot_encoder",OneHotEncoder(handle_unknown = 'ignore'),categorical_features)
])


modelRid= make_pipeline(my_col_trans, Ridge(alpha=10,random_state=0))
modelRid.fit(X_train,y_train)
st.write(modelRid.score(X_test,y_test))


titre = "Prédiction prix d'une maison"
original_title = '<p style="font-family:Courier; color:Blue; font-size: 42px;">{} </p>'.format(titre)
st.markdown(original_title, unsafe_allow_html=True)



caracteristique_maison = [0 for _ in range(17)]

nb_bedrooms = st.number_input(label="Nombre de chambre :",min_value=0,step=1, key="bedrooms")
caracteristique_maison[0] = nb_bedrooms

nb_bathrooms = st.number_input(label="Nombre de salle de bain :",min_value=0.,step=1.,format="%.2f", key="bathrooms")
caracteristique_maison[1] = nb_bathrooms

sqft_living = st.number_input(label="Espace interieur (en sqft) :",min_value=0, key="sqft_living")
caracteristique_maison[2]=sqft_living
caracteristique_maison[3]=sqft_living**2

sqft_lot  = st.number_input(label="Terrain (en sqft) :",min_value=0, key="sqft_lot")
caracteristique_maison[4] = sqft_lot

nb_floor = st.number_input(label="Nombre d'etage :",min_value=0.,step=1.,format="%.2f", key="floors")
caracteristique_maison[5] = nb_floor

waterfront = st.radio(
     "A t-elle vue sur la mer ",
     ('Oui', 'Non'))
if waterfront == 'Oui':
  caracteristique_maison[6] = 1
else:
  caracteristique_maison[6] = 0

view = st.number_input(label="Qualite de la vue (entre 0 = mauvaise et 4 = bonne) :",min_value=0,step=1,max_value=4, key="view")
caracteristique_maison[7] = view
  
condition = st.number_input(label="Condition (entre 0 et 4) :",min_value=0,step=1,max_value=4, key="condition")
caracteristique_maison[8] = condition

grade = st.number_input(label="Qualite de la construction (entre 1 et 13) :",min_value=1,step=1,max_value=13, key="grade")
caracteristique_maison[9] = grade

sqft_above = st.number_input(label="Espace interieur (en sqft) au dessus du niveau du sol :",min_value=0, key="sqft_above :")
caracteristique_maison[10] = sqft_above
caracteristique_maison[11] = sqft_above**2


sqft_basement = st.number_input(label="Espace interieur (en sqft) en dessous du niveau du sol :",min_value=0, key="sqft_basement :")
caracteristique_maison[12] = sqft_basement
caracteristique_maison[13] = sqft_basement**2

zip_code = st.number_input(label="zip code :",min_value=98000,max_value=99000, key="zip code :")
caracteristique_maison[14] = zip_code

sqft_living15 = st.number_input(label="Espace interieur (en sqft) des 15 voisins :",min_value=0, key="sqft_living15 :")
caracteristique_maison[15]=sqft_living15

sqft_lot15  = st.number_input(label="Terrain (en sqft) des 15 voisins :",min_value=0, key="sqft_lot15 :")
caracteristique_maison[16] = sqft_lot15




if(st.button("Valider")):
    
    
    predic = int(modelRid.predict(pd.DataFrame(np.array(caracteristique_maison).reshape(1, -1),columns=X_train.columns)))
    
    new_title = '<p style="font-family:sans-serif; color:Green;width:100%;text-align:center; font-size: 36px;">{}</p>'.format(predic)
    st.markdown(new_title, unsafe_allow_html=True)