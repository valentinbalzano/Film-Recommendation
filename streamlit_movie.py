import pandas as pd
import streamlit as st
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import unicodedata
from sklearn.preprocessing import StandardScaler
from PIL import Image
import time

# Logo webapp
st.image('./Capture d’écran 2021-11-19 à 10.59.57.png') #/Users/valentinbalzano/Documents/myproject/Capture d’écran 2021-11-19 à 10.59.57.png


# Title's Webapp
st.title('Moteur de recommandation de films en ligne')

# Download Data
df_final = pd.read_csv('./df_final.csv')#/Users/valentinbalzano/Desktop/Support/df_final.csv

#Drop Columns
df_final.drop(columns = ‘Unnamed: 0’, inplace = True)

# Modify title's column
df_final.rename(columns={'title_y': 'Films', 'primaryName': 'Réalisateurs', 'startYear': 'Sortie', 'genres_y': 'Catégories', 'averageRating_y': 'Notes', 'numVotes_y': 'Votes'}, inplace=True)

# modifier Dtype de la colonne startYear 
df_final['Sortie'] = df_final['Sortie'].apply(lambda x: int(x))

# algorithme de recommandation
X = df_final.drop(columns = ['titleId_x', 'titleType', 'primaryTitle', 'originalTitle', 'Sortie', 'endYear', 'runtimeMinutes', 'Catégories', 'titleId_y', 'ordering', 'Films', 'region', 'genres_uniques', 'genres2', 'Réalisateurs', 'decade'])
y = df_final['Films']



# Fonction pour supprimer accents et mettre la chaîne de caractères en minuscule
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


# Normalisation
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)



# Création dataframe avec les valeurs normalisées
df_scaled = pd.DataFrame(X_scaled, columns = ['Steven Spielberg', 'Joel Coen', 'Ethan Coen', 'Martin Scorsese',
       'Ridley Scott', 'Tim Burton', 'Quentin Tarantino', 'Ron Howard',
       'Stanley Kubrick', 'Richard Donner', 'Christopher Nolan',
       'Oliver Stone', 'Guy Ritchie', 'David Lynch', 'Peter Jackson',
       'Francis Ford Coppola', 'Alfred Hitchcock', 'Wilfred Jackson',
       'John Carpenter', 'Clint Eastwood', 'Danny Boyle', 'Hayao Miyazaki',
       'isAdult', 'Notes', 'Votes', 'Action', 'Adventure',
       'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
       'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller',
       'War', 'Western'])



# weight readjustment directors's column
liste = df_scaled.loc[:, 'Steven Spielberg':'Hayao Miyazaki'].columns
for i in liste:
    df_scaled[i] = df_scaled[i].apply(lambda x: x/4)



# weight readjustment Drama's column
df_scaled['Drama'] = df_scaled['Drama'].apply(lambda x: x/3)


nb_film = st.slider('Nombre de films à recommander', min_value = 1, max_value = 20)



# Fonction moteur de recommandation
def reco_films_std():
    i = 2

    # Création d'une variable x dans laquelle l'utilisateur input le films cible 
    name = remove_accents(st.text_input('', key = 1, placeholder = 'Recherche')) 
   
    # On propose une pré selection selon le mot clé entré
    df_pre_selection = df_final.loc[df_final['Films'].apply(lambda x: remove_accents(x)).str.contains(name), ['Films', 'Catégories', 'Réalisateurs','Notes', 'Votes', 'Sortie', 'decade']]
   
    # Tant que la pré selection n'affiche pas de films, on demande à l'utilisateur de rentrer un autre mot clé
    if len(df_pre_selection) == 0:
    	st.warning('Vous avez saisi un films de merde. Essayez à nouveau.')
    	name = remove_accents(st.text_input('', key = i, placeholder = 'Recherche'))
    	df_pre_selection = df_final.loc[df_final['Films'].apply(lambda x: remove_accents(x)).str.contains(name), ['Films', 'Catégories', 'Réalisateurs','Notes', 'Votes', 'Sortie', 'decade']]
    	i += 1

    else:
        # Condition : si la pré selection contien une seule ligne, alors la pré-selection ne s'affiche pas sinon, elle s'affiche
    	if len(df_pre_selection.index) > 1:

       		# 1ere condition : la pré selection s'affiche
        	df_pre_selection['Sortie'] = df_pre_selection['Sortie'].apply(lambda x: int(x))
        	st.dataframe(df_pre_selection[['Films', 'Catégories', 'Réalisateurs', 'Notes', 'Votes', 'Sortie']])
       
       		# On demande à l'utilisateur de rentrer le # de la ligne qui correspond au film recherché
        	y = st.number_input('A quel film pensez vous?', help = ' Entrez le numéro de la ligne', value = 1, step = 1)
      
        	# On stock l'index du films correspondant dans la variable z
        	z = df_pre_selection.iloc[y-1:y].index[0]
        
        	# On utilise l'index de notre variable z pour faire tourner notre algorithm de recommandation
        	# On fit notre algorithm avec la méthode des voisins les plus proches (KNN)
        	distanceKNN = NearestNeighbors(n_neighbors=nb_film+1).fit(X_scaled)
        
        	# On affiche le résultat de notre algorithm. Un array dans lequel on trouve la distance entre le films cible / les films les plus proches et l'index des films correspondants
        	df = distanceKNN.kneighbors(df_scaled.loc[df_scaled.index == z, df_scaled.columns])
        
        	# On se sert de cet array pour isoler les index des films les plus proches du films cible et les afficher avec un certain nombre d'informations 
        	df_final_2 = df_final.iloc[df[1][0][1:]]
        	st.subheader('Recommandations')
        	return st.dataframe(df_final_2[['Films', 'Catégories', 'Réalisateurs', 'Notes', 'Votes', 'Sortie']])
    	else:

        	# idem que précédemment sans l'affichage de la pré selection
        	st.dataframe(df_pre_selection[['Films', 'Catégories', 'Réalisateurs', 'Notes', 'Votes', 'Sortie']])
        	z = df_pre_selection.index[0]
        	distanceKNN = NearestNeighbors(n_neighbors=nb_film+1).fit(X_scaled)
        	df = distanceKNN.kneighbors(df_scaled.loc[df_scaled.index == z, df_scaled.columns])
        	df_final_2 = df_final.iloc[df[1][0][1:]]
        	st.subheader('Recommandations')
        	return st.dataframe(df_final_2[['Films', 'Catégories', 'Réalisateurs', 'Notes', 'Votes', 'Sortie']])



with st.spinner('Wait for it...'):
	time.sleep(1)



reco_films_std()
