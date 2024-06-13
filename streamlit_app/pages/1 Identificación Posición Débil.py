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
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season!='2023-2024']
    return df
df = load_data()

st.title("Identifiación posición débil")

st.write("Puedes usar esta página para identificar que posición tienes más debil respecto al resto de equipos en las estadísticas que te interesn según el perfil de tu equipo.")

# Supongamos que 'df' es tu DataFrame
teams = df['team'].unique()
st.write("A continuación, seleccione el equipo que quiere analizar respecto al resto:")
selected_team = st.selectbox('Selecciona un equipo:', teams)
st.session_state.selected_team = selected_team

st.write("Una vez seleccionado el equipo, seleccione las estadisticas a analizar y filtre según le interese para su análisis.")
selected_stat1 = st.selectbox("Selecciona la primera estadística", df.columns)
selected_stat2 = st.selectbox("Selecciona la segunda estadística", df.columns.drop(selected_stat1))
filtered_df = pd.DataFrame()
positions = df['pos'].unique()
selected_positions = st.multiselect("Selecciona las posiciones", positions)
if selected_positions:    
    filtered_df = df[df['pos'].isin(selected_positions)]

    seasons = filtered_df['season'].unique()
    selected_season = st.selectbox("Selecciona una temporada", seasons)
    if selected_season:
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
        
        leagues = filtered_df['league'].unique()
        selected_leagues = st.multiselect("Selecciona las ligas", leagues)
        if selected_leagues:
            filtered_df = filtered_df[filtered_df['league'].isin(selected_leagues)]
            
            teams = filtered_df[filtered_df['team']!=selected_team]['team'].unique()
            selected_teams = st.multiselect("Selecciona los equipos", teams)
            if selected_teams:
                filtered_df = filtered_df[filtered_df['team'].isin(selected_teams)]

                min_age, max_age = st.slider("Selecciona rango de edad", int(filtered_df['age'].min()), int(filtered_df['age'].max()), (int(filtered_df['age'].min()), int(filtered_df['age'].max())))
                filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

                min_minutes, max_minutes = st.slider("Selecciona el rango de porcentjae de minutos", int(filtered_df['Min%'].min()), int(filtered_df['Min%'].max()), (int(filtered_df['Min%'].min()), int(filtered_df['Min%'].max())))
                filtered_df = filtered_df[(filtered_df['Min%'] >= min_minutes) & (filtered_df['age'] <= max_minutes)]

if len(filtered_df) > 0:
    team_data = df[(df['team'] == selected_team) & (df.season == selected_season) & (df.pos.isin(selected_positions))]
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
