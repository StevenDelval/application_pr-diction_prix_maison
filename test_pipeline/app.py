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
       'sqft_living', 'sqft_living_carre', 'sqft_lot','sqft_above',
       'sqft_basement','sqft_living15', 'sqft_lot15'
       ]
categorical_features = [
       'floors','waterfront','view', 'condition', 'grade','zipcode'
       ]

my_col_trans = ColumnTransformer([
    ("sclal",StandardScaler(),numerical_features),
    ("pol",PolynomialFeatures(2),numerical_features),
    ("one_hot_encoder",OneHotEncoder(handle_unknown = 'ignore'),categorical_features)
])


modelRid= make_pipeline(my_col_trans, Ridge(alpha=10))
modelRid.fit(X_train,y_train)
modelRid.score(X_train,y_train)
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
    
    
    predic = int(modelRid.predict(pd.DataFrame(np.array([3,2.0,1510,2280100,4560,2.0,0,0,4,7,1510,2280100,0,0,98116,1990,5000]).reshape(1, -1),columns=X_train.columns)))
    
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 24px;">{}</p>'.format(predic)
    st.markdown(new_title, unsafe_allow_html=True)