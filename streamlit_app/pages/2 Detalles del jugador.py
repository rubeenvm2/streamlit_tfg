import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(
page_title="Detalles del jugador",
page_icon="🔎",
layout="wide",
initial_sidebar_state="expanded")
# CSS para estilizar la tabla



# Cargar los datos
@st.cache_data
def load_data():
    # Cargar tu conjunto de datos de jugadores
    return pd.read_csv('streamlit_app/data.csv')

def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

df = load_data()
try:
    
    team_of_player = st.session_state.selected_team
    st.title(f"Detalles del jugador")
    st.write(f"Escoge un jugador de tu equipo ({team_of_player}) para analizar y compararlo posteriormente frente a otros jugadores.")
    last_season_df = df[(df.season=='2022-2023') & (df.team == team_of_player)].drop('season', axis=1)
    selected_player = st.selectbox("Escoge un jugador", last_season_df['player'].unique())
    player = last_season_df[(last_season_df['player'] == selected_player) & (last_season_df['team'] == team_of_player)]
    st.header("Player bio")
    # Información de la biografía del jugador
    player_name = player['player'].values[0]
    player_position = player['pos'].values[0]
    player_age = player['age'].values[0]
    player_nation = player['nation'].values[0]
    player_starts = player['Starts'].values[0]
    if selected_player != "":
        css = """
        <style>
            .bio-grid {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              grid-gap: 5px;
              margin-bottom: 20px; /* Agregando margen después de la grid */
            }
            .bio-stat-heading {
              font-weight: bold;
              color: #888;
              margin-bottom: 4px;
            }
            *, ::after, ::before {
              box-sizing: border-box;
            }
            .player-bio {
              color: #FFFFFF;
            }
            .player-stats {
              color: white;
              font-family: Arial, sans-serif;
            }
            .row {
              --bs-gutter-x: 1.5rem;
              --bs-gutter-y: 0;
            }
            body {
              font-family: var(--bs-body-font-family);
              font-size: var(--bs-body-font-size);
              font-weight: var(--bs-body-font-weight);
              line-height: var(--bs-body-line-height);
              color: var(--bs-body-color);
              text-align: var(--bs-body-text-align);
              -webkit-text-size-adjust: 100%;
            }
            :root {
              --bs-blue: #0d6efd;
              --bs-indigo: #6610f2;
              --bs-purple: #6f42c1;
              --bs-pink: #d63384;
              --bs-red: #dc3545;
              --bs-orange: #fd7e14;
              --bs-yellow: #ffc107;
              --bs-green: #198754;
              --bs-teal: #20c997;
              --bs-cyan: #0dcaf0;
              --bs-white: #fff;
              --bs-gray: #6c757d;
              --bs-gray-dark: #343a40;
              --bs-gray-100: #f8f9fa;
              --bs-gray-200: #e9ecef;
              --bs-gray-300: #dee2e6;
              --bs-gray-400: #ced4da;
              --bs-gray-500: #adb5bd;
              --bs-gray-600: #6c757d;
              --bs-gray-700: #495057;
              --bs-gray-800: #343a40;
              --bs-gray-900: #212529;
              --bs-primary: #0d6efd;
              --bs-secondary: #6c757d;
              --bs-success: #198754;
              --bs-info: #0dcaf0;
              --bs-warning: #ffc107;
              --bs-danger: #dc3545;
              --bs-light: #f8f9fa;
              --bs-dark: #212529;
              --bs-primary-rgb: 13,110,253;
              --bs-secondary-rgb: 108,117,125;
              --bs-success-rgb: 25,135,84;
              --bs-info-rgb: 13,202,240;
              --bs-warning-rgb: 255,193,7;
              --bs-danger-rgb: 220,53,69;
              --bs-light-rgb: 248,249,250;
              --bs-dark-rgb: 33,37,41;
              --bs-white-rgb: 255,255,255;
              --bs-black-rgb: 0,0,0;
              --bs-body-color-rgb: 33,37,41;
              --bs-body-bg-rgb: 255,255,255;
              --bs-font-sans-serif: system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","Liberation Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
              --bs-font-monospace: SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;
              --bs-gradient: linear-gradient(180deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0));
              --bs-body-font-family: var(--bs-font-sans-serif);
              --bs-body-font-size: 1rem;
              --bs-body-font-weight: 400;
              --bs-body-line-height: 1.5;
              --bs-body-color: #212529;
              --bs-body-bg: #fff;
            }
        </style>
        """
        # HTML para mostrar los datos del jugador
        html = f"""
        {css}
        <div class="bio-grid">
            <div class="bio-stat">
                <div class="bio-stat-heading">Name</div>
                <div class="bio-stat-value" id="player-name"><span class="white-text">{player_name}</span></div>
            </div>
            <div class="bio-stat">
                <div class="bio-stat-heading">Position</div>
                <div class="bio-stat-value" id="player-position"><span class="white-text">{player_position}</span></div>
            </div>
            <div class="bio-stat">
                <div class="bio-stat-heading">Age</div>
                <div class="bio-stat-value" id="player-age"><span class="white-text">{player_age}</span></div>
            </div>
            <div class="bio-stat">
                <div class="bio-stat-heading">Nation</div>
                <div class="bio-stat-value" id="player-nation"><span class="white-text">{player_nation}</span></div>
            </div>
            <div class="bio-stat">
                <div class="bio-stat-heading">Starts</div>
                <div class="bio-stat-value" id="player-starts"><span class="white-text">{player_starts}</span></div>
            </div>
        </div>
        """
        
        # Mostrar la tabla de biografía del jugador en Streamlit
        st.markdown(html, unsafe_allow_html=True)
        
        selected_features = st.multiselect('Seleccionar características clave:', df.select_dtypes(include=np.number).columns)
        #selected_features = ['xG', 'nxG']  # Puedes agregar más estadísticas aquí
    if len(selected_features) > 0:
        position_data = df[df['pos'] == player_position][selected_features]
        player_data = df[df['player'] == selected_player]
        player_position = player_data['pos'].values[0]
        
        st.header("Estadisticas clave")
        st.write(f"La barra que puedes observar a continuación tiene como valores minimo y máximo cogiendo únicamente los jugadores de la posición de {player_name}.")
    
        for feature in selected_features:
            min_val = df[df['pos'] == player_position][feature].min()
            max_val = df[df['pos'] == player_position][feature].max()
            player_val = player_data[feature].values[0]
      
            # Calcular el porcentaje del valor del jugador respecto al máximo
            percentage = (player_val - min_val) / (max_val - min_val) * 100
    
            # Determinar el color basado en el porcentaje (de tonos oscuros a tonos claros de verde)
            green_color = int(255 - (percentage * 1.27))
            red_color = int(percentage * 1.27)
            color = f"rgb({red_color}, {green_color}, 0)"
            # Mostrar la barra de progreso con el estilo personalizado y el percentil
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="margin-bottom: 5px;">{feature}: {player_val} (Percentil: {percentage:.2f}%)</div>
                <div style="display: flex; align-items: center;">
                    <div style="margin-right: 10px;">{min_val:.2f}</div>
                    <div style="position: relative; height: 30px; width: 100%; border-radius: 5px; background: #f0f0f0;">
                        <div style="width: {percentage}%; height: 100%; border-radius: 5px; background: {color};"></div>
                    </div>
                    <div style="margin-left: 10px;">{max_val:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("Clica el siguiente botón si quieres buscar jugadores similares a este.")
    numeric_features = df[df.columns].select_dtypes(include=[np.number]).columns.tolist()
    df_normalized = normalize_features(df.copy(), numeric_features)
    st.session_state.player = df_normalized.loc[player.index[0]]
    st.page_link("pages/3_Análisis_de_similitud.py", label="Botón")
except:
    st.write("Selecciona a un equipo en la pestaña de identificación posición débil antes de analizar a un jugador.")



