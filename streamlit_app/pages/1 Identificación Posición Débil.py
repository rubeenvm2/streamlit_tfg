import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
page_title="Identifiación posición débil",
page_icon="📉",
layout="wide",
initial_sidebar_state="expanded")

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df[df.season!='2023-2024']
    return df
df = load_data()

st.title("Identifiación posición débil")

st.write("Puedes usar esta página para identificar que posición tienes más debil respecto al resto de equipos en las estadísticas que te interesn según el perfil de tu equipo.")

# Supongamos que 'df' es tu DataFrame
teams = df['team'].unique()

selected_team = st.selectbox('Selecciona tu equipo:', teams)
st.session_state.selected_team = selected_team

selected_stat1 = st.selectbox("Selecciona la primera estadística", df.columns)  # Cambia 6: por el índice de tu primera estadística
selected_stat2 = st.selectbox("Selecciona la segunda estadística", df.columns.drop(selected_stat1))

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
    fig = go.Figure()
    
    # Scatter plot para los datos del equipo seleccionado
    fig.add_trace(go.Scatter(
        x=team_data[selected_stat1],
        y=team_data[selected_stat2],
        mode='markers+text',
        text=team_data['player'],
        textposition='top center',
        marker=dict(color='orange', size=10),
        name=selected_team
    ))

    # Scatter plot para los datos filtrados
    fig.add_trace(go.Scatter(
        x=filtered_df[selected_stat1],
        y=filtered_df[selected_stat2],
        mode='markers',
        marker=dict(color='white', opacity=0.6),
        name='Otros equipos'
    ))

    # Configurar las etiquetas y el título del gráfico
    fig.update_layout(
        title=f'{selected_stat1} vs {selected_stat2}',
        xaxis_title=selected_stat1,
        yaxis_title=selected_stat2,
        legend_title='Leyenda'
    )

    st.plotly_chart(fig)