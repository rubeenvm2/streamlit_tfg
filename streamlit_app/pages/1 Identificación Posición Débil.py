import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


st.set_page_config(
page_title="Identifiación posición débil",
page_icon="📉",
layout="wide",
initial_sidebar_state="expanded")

# Cargar los datos
@st.cache_data
def load_data():
    # Cargar tu conjunto de datos de jugadores
    return pd.read_csv('streamlit_app/data.csv')

df = load_data()

st.title("Identifiación posición débil")

st.write("Puedes usar esta página para identificar que posición tienes más debil respecto al resto de equipos en las estadísticas que te interesn según el perfil de tu equipo.")

teams = df['team'].unique()

selected_team = st.selectbox('Selecciona tu equipo:', teams)
st.session_state.selected_team = selected_team

selected_stat1 = st.selectbox("Selecciona la segunda estadística", df.columns)  # Cambia 6: por el índice de tu primera estadística
selected_stat2 = st.selectbox("Selecciona la primera estadística", df.columns)

positions = df['pos'].unique()
selected_positions = st.multiselect("Selecciona las posiciones a mostrar", positions)

seasons = df['season'].unique()
selected_season = st.selectbox("Selecciona la temporada que quieres mirar", seasons)

leagues = df['league'].unique()
selected_league = st.multiselect("Selecciona las ligas deseadas en el gráfico", leagues)

min_age, max_age = st.slider("Selecciona rango de edad", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
min_minutes, max_minutes = st.slider("Selecciona el rango de porcentjae de minutos", int(df['Min%'].min()), int(df['Min%'].max()), (int(df['Min%'].min()), int(df['Min%'].max())))

filtered_df = df[
    (df['pos'].isin(selected_positions)) &
    (df['season'] == selected_season) &
    (df['league'].isin(selected_league)) &
    (df['age'] >= min_age) & (df['age'] <= max_age) &
    (df['Min%'] >= min_minutes) & (df['Min%'] <= max_minutes) &
    (df['team']!=(selected_team))
]

if len(filtered_df) > 0:
    team_data = df[(df['team'] == selected_team) & (df.season == selected_season)]

    plt.scatter(x=filtered_df[selected_stat1], y=filtered_df[selected_stat2], color='black', alpha=0.3)
    plt.scatter(team_data[selected_stat1], team_data[selected_stat2], color='orange', label=f'{selected_team}', s=100, alpha=0.8)

    for x, y, player in zip(team_data[selected_stat1], team_data[selected_stat2], team_data['player']):
            # Calcular una ligera variación en la posición de la anotación
            sign = np.random.choice([-1, 1])
            offset = sign * (team_data.shape[0] / 30)  # Ajusta el valor 30 según la densidad de puntos
            plt.annotate(player, (x, y), textcoords="offset points", xytext=(5+offset,-10+offset), ha='left')
    plt.xlabel(selected_stat1)
    plt.ylabel(selected_stat2)
    plt.title(f"{selected_stat1} vs {selected_stat2}")
    plt.legend()
    st.pyplot(plt)
