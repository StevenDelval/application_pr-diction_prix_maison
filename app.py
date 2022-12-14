import streamlit as st
import pandas as pd
import numpy as np
import pickle

pickle_in = open('modelRid.pkl', 'rb') 
modelRid = pickle.load(pickle_in) 

titre = "Prédiction prix d'une maison"
original_title = '<p style="font-family:Courier; color:Blue; font-size: 42px;">{} </p>'.format(titre)
st.markdown(original_title, unsafe_allow_html=True)

columns_model = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_living_carre', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_above_carre', 'sqft_basement', 'sqft_basement_carre', 'zipcode',
       'sqft_living15', 'sqft_lot15']

caracteristique_maison = [0 for _ in range(17)]

nb_bedrooms = st.number_input(label="Nombre de chambre :",min_value=0,step=1, value=4, key="bedrooms")
caracteristique_maison[0] = nb_bedrooms

nb_bathrooms = st.number_input(label="Nombre de salle de bain :",value=2.0,min_value=0.,step=1.,format="%.2f", key="bathrooms")
caracteristique_maison[1] = nb_bathrooms

sqft_living = st.number_input(label="Espace interieur (en sqft) :",value=2230, min_value=0, key="sqft_living")
caracteristique_maison[2]=sqft_living
caracteristique_maison[3]=sqft_living**2

sqft_lot  = st.number_input(label="Terrain (en sqft) :",value=5000,min_value=0, key="sqft_lot")
caracteristique_maison[4] = sqft_lot

nb_floor = st.number_input(label="Nombre d'etage :",value=1.5,min_value=0.,step=1.,format="%.2f", key="floors")
caracteristique_maison[5] = nb_floor

waterfront = st.radio(
     "A t-elle vue sur la mer ?",
     ('Non','Oui' ))
if waterfront == 'Oui':
  caracteristique_maison[6] = 1
else:
  caracteristique_maison[6] = 0

view = st.number_input(label="Qualite de la vue (entre 0 = mauvaise et 4 = bonne) :",value=0,min_value=0,step=1,max_value=4, key="view")
caracteristique_maison[7] = view
  
condition = st.number_input(label="Condition (entre 0 et 4) :",value=4,min_value=0,step=1,max_value=4, key="condition")
caracteristique_maison[8] = condition

grade = st.number_input(label="Qualite de la construction (entre 1 et 13) :",value=7, min_value=1,step=1,max_value=13, key="grade")
caracteristique_maison[9] = grade

sqft_above = st.number_input(label="Espace interieur (en sqft) au dessus du niveau du sol :",value=1510, min_value=0, key="sqft_above :")
caracteristique_maison[10] = sqft_above
caracteristique_maison[11] = sqft_above**2


sqft_basement = st.number_input(label="Espace interieur (en sqft) en dessous du niveau du sol :",value=720,min_value=0, key="sqft_basement :")
caracteristique_maison[12] = sqft_basement
caracteristique_maison[13] = sqft_basement**2

liste_zip_code= [98144, 98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010,
       98011, 98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029,
       98030, 98031, 98032, 98033, 98034, 98038, 98039, 98040, 98042,
       98045, 98052, 98053, 98055, 98056, 98058, 98059, 98065, 98070,
       98072, 98074, 98075, 98077, 98092, 98102, 98103, 98105, 98106,
       98107, 98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119,
       98122, 98125, 98126, 98133, 98136, 98146, 98148, 98155,
       98166, 98168, 98177, 98178, 98188, 98198, 98199]

zip_code = st.selectbox("Zip Code: ", 
                     liste_zip_code,)

caracteristique_maison[14] = zip_code

sqft_living15 = st.number_input(label="Espace interieur (en sqft) des 15 voisins :",value=2140,min_value=0, key="sqft_living15 :")
caracteristique_maison[15]=sqft_living15

sqft_lot15  = st.number_input(label="Terrain (en sqft) des 15 voisins :",value=5000,min_value=0, key="sqft_lot15 :")
caracteristique_maison[16] = sqft_lot15




if(st.button("Valider")):
    
    predic = int(modelRid.predict(pd.DataFrame(np.array(caracteristique_maison).reshape(1, -1),columns=columns_model)))
    
    new_title = '<p style="font-family:sans-serif; color:Green;width:100%;text-align:center; font-size: 36px;">{} $</p>'.format(predic)
    st.markdown(new_title, unsafe_allow_html=True)