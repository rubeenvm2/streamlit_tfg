import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt

st.set_page_config(
page_title="Predicción de rendimiento",
page_icon="📈",
layout="wide",
initial_sidebar_state="expanded")
# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df_no_dav = pd.read_csv('model_output_no_dav.csv')
    df_dav = pd.read_csv('model_output_dav.csv')
    return df,df_no_dav,df_dav

def stateful_button(*args, key=None, sidebar=False, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False
    if sidebar==False:
        if st.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]
    else:
        if st.sidebar.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]
    return st.session_state[key]

def create_scatterplots(df, player):
    all_seasons = sorted(df['season'].unique(), key=lambda x: int(x.split('-')[0]))
    fig, ax = plt.subplots()
    player = df[df.player == player]
    # Crear un gráfico de dispersión de las temporadas vs. la característica seleccionada
    ax.scatter(player['season'], player['DAVIES'], color='blue', s=100, alpha=0.8, label='Jugador')

    # Añadir las etiquetas de los puntos en el gráfico con formato de dos decimales
    for x, y in zip(player['season'], player['DAVIES']):
        ax.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

    # Calcular y añadir las medianas por temporada y posición
    relevant_seasons = player['season'].unique()
    medianas = df[df['season'].isin(relevant_seasons)].groupby('season')['DAVIES'].mean().reset_index()
    medianas['season'] = pd.Categorical(medianas['season'], categories=all_seasons, ordered=True)

    ax.scatter(medianas['season'], medianas['DAVIES'], color='red', s=100, alpha=0.8, label='Media de su posición')

    # Añadir las etiquetas de los puntos medianos en el gráfico
    for x, y in zip(medianas['season'], medianas['DAVIES']):
        ax.text(x, y, f'{y:.2f}', fontsize=9, ha='left')

    # Configurar las etiquetas y el título del gráfico
    ax.set_xlabel('Season')
    ax.set_ylabel('DAVIES')
    ax.set_title(f'Histórico de {player.player.unique()[0]} - DAVIES', fontsize='small')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

st.title("Predicción de rendimiento")

df, df_no_dav, df_dav = load_data()
similar_player = st.session_state.similar_player
similar_team = st.session_state.similar_team
st.write(f"En esta página primeramente verás una estadística de la media de rendimiento para cada temporada de los jugadores existentes en la base de datos. Estas puntuaciones van de 0 a infinito y han sido predichas mediante un modelo de machine learning basandose en todo tipo de estadisticas, tanto defensivas como ofensivas, como de creación de juego, progresión con el balón. Captando así el estilo de juego de {similar_player} para predecir correctamente el rendimiento.")
create_scatterplots(df, similar_player)
if similar_team:
    actual_value = df_dav[(df_dav.player == similar_player) & (df_dav.season == df_dav.season.max()) & (df_dav.team == similar_team)]['DAVIES'].unique()[0]
    col1,col2 = st.columns(2)
    with col1:
        if stateful_button(f'Predecir rendimiento de {similar_player} en la temporada siguiente a la actual. (con DAVIES)', key="pred_DAV"):
            prediction_value = df_dav[(df_dav.player == similar_player) & (df_dav.season == df_dav.season.max()) & (df_dav.team == similar_team)]['DAVIES_next_season'].unique()[0]
            st.write(f"La performance de este año del jugador ha sido de {actual_value:.2f}. La predicción para el año siguiente es {prediction_value:.2f}")

    with col2:
        if stateful_button(f'Predecir rendimiento de {similar_player} en la temporada siguiente a la actual. (sin DAVIES)', key="pred_no_DAV"):
            prediction_value = df_no_dav[(df_no_dav.player == similar_player) & (df_no_dav.season == df_no_dav.season.max()) & (df_no_dav.team == similar_team)]['DAVIES_next_season'].unique()[0]
            st.write(f"La performance de este año del jugador ha sido de {actual_value:.2f}. La predicción para el año siguiente es {prediction_value:.2f}")