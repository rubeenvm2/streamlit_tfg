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


# Diccionario de mapeo de posiciones
posiciones = {
    'FW': 'Delantero',
    'MF': 'Centrocampista',
    'DF': 'Defensa',
    'FW,MF': 'Delantero-Centrocampista',
    'MF,FW': 'Centrocampista-Delantero',
    'FW,DF': 'Delantero-Defensa',
    'DF,FW': 'Defensa-Delantero',
    'MF,DF': 'Centrocampista-Defensa',
    'DF,MF': 'Defensa-Centrocampista'
    # Añade más posiciones según sea necesario
}

ligas = {
    'espla liga': 'La Liga (ESP)',
    'gerbundesliga': 'Bundesliga (GER)',
    'engpremier league': 'Premier League (ENG)',
    'itaserie a': 'Serie A (ITA)',
    'fraligue ': 'Ligue 1 (FRA)'
    # Añade más posiciones según sea necesario
}

# Diccionario de mapeo de estadísticas
estadisticas = {
    'groundDuelsWon': 'Duelos terrestres ganados',
    'groundDuelsWonPercentage': 'Porcentaje de duelos terrestres ganados',
    'aerialDuelsWonPercentage': 'Porcentaje de duelos aéreos ganados (SofaScore)',
    'wasFouled': 'Faltas recibidas (SofaScore)',
    'dispossessed': 'Pérdidas de balón',
    'accurateFinalThirdPasses': 'Pases precisos en el último tercio',
    'bigChancesCreated': 'Grandes oportunidades creadas',
    'keyPasses': 'Pases clave (SofaScore)',
    'accurateCrosses': 'Centros precisos',
    'accurateCrossesPercentage': 'Porcentaje de centros precisos',
    'accurateLongBalls': 'Balones largos precisos',
    'accurateLongBallsPercentage': 'Porcentaje de balones largos precisos',
    'dribbledPast': 'Regateado',
    'bigChancesMissed': 'Grandes oportunidades falladas',
    'hitWoodwork': 'Tiros al poste',
    'errorLeadToGoal': 'Errores que llevaron a gol',
    'errorLeadToShot': 'Errores que llevaron a tiro',
    'passToAssist': 'Pase para asistencia',
    'player': 'Jugador',
    'team': 'Equipo',
    'season': 'Temporada',
    'league': 'Liga',
    'nation': 'Nación',
    'pos': 'Posición',
    'age': 'Edad',
    'born': 'Nacimiento',
    'MP': 'Partidos jugados',
    'Starts': 'Titularidades',
    'Min': 'Minutos jugados',
    '90s': 'Partidos completos (90 minutos)',
    'Gls': 'Goles',
    'Ast': 'Asistencias',
    'G+A': 'Goles + Asistencias',
    'G-PK': 'Goles sin penales',
    'PK': 'Penales marcados',
    'PKatt': 'Penales intentados',
    'CrdY': 'Tarjetas amarillas',
    'CrdR': 'Tarjetas rojas',
    'xG': 'Goles esperados (xG)',
    'npxG': 'Goles esperados sin penales (npxG)',
    'xAG': 'Asistencias esperadas (xAG)',
    'npxG+xAG': 'Goles esperados sin penales + Asistencias esperadas (npxG+xAG)',
    'PrgC': 'Pases progresivos completados',
    'PrgP': 'Pases progresivos',
    'PrgR': 'Carreras progresivas',
    'Gls_90': 'Goles por 90 minutos',
    'Ast_90': 'Asistencias por 90 minutos',
    'G+A_90': 'Goles + Asistencias por 90 minutos',
    'G-PK_90': 'Goles sin penales por 90 minutos',
    'G+A-PK_90': 'Goles + Asistencias sin penales por 90 minutos',
    'xG_90': 'Goles esperados por 90 minutos (xG)',
    'xAG_90': 'Asistencias esperadas por 90 minutos (xAG)',
    'xG+xAG_90': 'Goles esperados + Asistencias esperadas por 90 minutos (xG+xAG)',
    'npxG_90': 'Goles esperados sin penales por 90 minutos (npxG)',
    'npxG+xAG_90': 'Goles esperados sin penales + Asistencias esperadas por 90 minutos (npxG+xAG)',
    'Sh': 'Tiros',
    'SoT': 'Tiros a puerta',
    'SoT%': 'Porcentaje de tiros a puerta',
    'Sh/90': 'Tiros por 90 minutos',
    'SoT/90': 'Tiros a puerta por 90 minutos',
    'G/Sh': 'Goles por tiro',
    'G/SoT': 'Goles por tiro a puerta',
    'Dist': 'Distancia media de los tiros',
    'FK': 'Tiros libres',
    'npxG/Sh': 'Goles esperados sin penales por tiro (npxG/Sh)',
    'G-xG': 'Diferencia entre goles y goles esperados (G-xG)',
    'np:G-xG': 'Diferencia entre goles sin penales y goles esperados sin penales (np:G-xG)',
    'Total_Cmp': 'Pases completados',
    'Total_Att': 'Pases totales intentados',
    'Total_Cmp%': 'Porcentaje de pases completados',
    'Total_TotDist': 'Distancia total de pases',
    'Total_PrgDist': 'Distancia progresiva de pases',
    'Short_Cmp': 'Pases cortos completados',
    'Short_Att': 'Pases cortos intentados',
    'Short_Cmp%': 'Porcentaje de pases cortos completados',
    'Medium_Cmp': 'Pases medios completados',
    'Medium_Att': 'Pases medios intentados',
    'Medium_Cmp%': 'Porcentaje de pases medios completados',
    'Long_Cmp': 'Pases largos completados',
    'Long_Att': 'Pases largos intentados',
    'Long_Cmp%': 'Porcentaje de pases largos completados',
    'xA': 'Asistencias esperadas (xA)',
    'A-xAG': 'Diferencia entre asistencias y asistencias esperadas (A-xAG)',
    'KP': 'Pases clave (FBref)',
    '1/3': 'Pases al último tercio',
    'PPA': 'Pases al área penal',
    'CrsPA': 'Centros al área penal',
    'Att': 'Pases intentados',
    'Pass Types_Live': 'Pases en juego',
    'Pass Types_Dead': 'Pases de balón parado',
    'Pass Types_FK': 'Pases de tiros libres',
    'Pass Types_TB': 'Pases en profundidad',
    'Pass Types_Sw': 'Cambios de juego',
    'Pass Types_Crs': 'Centros',
    'Pass Types_TI': 'Saques de banda',
    'Pass Types_CK': 'Corners',
    'Corner Kicks_In': 'Corners al primer palo',
    'Corner Kicks_Out': 'Corners al segundo palo',
    'Corner Kicks_Str': 'Corners al centro',
    'Outcomes_Cmp': 'Acciones completadas',
    'Outcomes_Off': 'Acciones ofensivas',
    'Outcomes_Blocks': 'Acciones bloqueadas',
    'SCA': 'Acciones que llevan a disparo (SCA)',
    'SCA90': 'Acciones que llevan a disparo por 90 minutos (SCA90)',
    'SCA_PassLive': 'Acciones que llevan a disparo en juego',
    'SCA_PassDead': 'Acciones que llevan a disparo de balón parado',
    'SCA_TO': 'Acciones que llevan a disparo por pérdida',
    'SCA_Sh': 'Acciones que llevan a disparo por disparo',
    'SCA_Fld': 'Acciones que llevan a disparo por falta',
    'SCA_Def': 'Acciones que llevan a disparo por defensa',
    'GCA': 'Acciones que llevan a gol (GCA)',
    'GCA90': 'Acciones que llevan a gol por 90 minutos (GCA90)',
    'GCA_PassLive': 'Acciones que llevan a gol en juego',
    'GCA_PassDead': 'Acciones que llevan a gol de balón parado',
    'GCA_TO': 'Acciones que llevan a gol por pérdida',
    'GCA_Sh': 'Acciones que llevan a gol por disparo',
    'GCA_Fld': 'Acciones que llevan a gol por falta',
    'GCA_Def': 'Acciones que llevan a gol por defensa',
    'Tkl': 'Entradas',
    'TklW': 'Entradas ganadas',
    'Tackles_Def 3rd': 'Entradas en el tercio defensivo',
    'Tackles_Mid 3rd': 'Entradas en el tercio medio',
    'Tackles_Att 3rd': 'Entradas en el tercio ofensivo',
    'Chall_Tkl': 'Desafíos ganados',
    'Chall_Att': 'Desafíos intentados',
    'Chall_Tkl%': 'Porcentaje de desafíos ganados',
    'Chall_Lost': 'Desafíos perdidos',
    'Blocks': 'Bloqueos',
    'Blocks_Sh': 'Bloqueos de disparos',
    'Blocks_Pass': 'Bloqueos de pases',
    'Int': 'Intercepciones',
    'Tkl+Int': 'Entradas + Intercepciones',
    'Clr': 'Despejes',
    'Err': 'Errores',
    'Touches': 'Toques',
    'Touches_Def Pen': 'Toques en el área defensiva',
    'Touches_Def 3rd': 'Toques en el tercio defensivo',
    'Touches_Mid 3rd': 'Toques en el tercio medio',
    'Touches_Att 3rd': 'Toques en el tercio ofensivo',
    'Touches_Att Pen': 'Toques en el área ofensiva',
    'Touches_Live': 'Toques en juego',
    'Take-Ons_Att': 'Regates intentados',
    'Take-Ons_Succ': 'Regates exitosos',
    'Take-Ons_Succ%': 'Porcentaje de regates exitosos',
    'Take-Ons_Tkld': 'Regates fallidos',
    'Take-Ons_Tkld%': 'Porcentaje de regates fallidos',
    'Carries': 'Conducciones',
    'Carries_TotDist': 'Distancia total conducida',
    'Carries_PrgDist': 'Distancia progresiva conducida',
    'Carries_PrgC': 'Conducciones progresivas',
    'Carries_1/3': 'Conducciones al último tercio',
    'Carries_CPA': 'Conducciones al área penal',
    'Carries_Mis': 'Conducciones fallidas',
    'Carries_Dis': 'Conducciones perdidas',
    'Receiving_Rec': 'Recepciones',
    'Receiving_PrgR': 'Recepciones progresivas',
    'Mn/MP': 'Minutos por partido',
    'Min%': 'Porcentaje de minutos jugados',
    'Compl': 'Partidos completos',
    'Subs': 'Sustituciones',
    'unSub': 'No sustituido',
    'Team_Succ_PPM': 'Puntos por partido del equipo',
    'Team_Succ_onG': 'Goles a favor del equipo',
    'Team_Succ_onGA': 'Goles en contra del equipo',
    'Team_Succ_+/-': 'Diferencia de goles del equipo',
    'Team_Succ_+/-90': 'Diferencia de goles por 90 minutos',
    'Team_Succ_On-Off': 'Eficacia del equipo con/sin jugador',
    'Team_Succ_onxG': 'Goles esperados a favor del equipo',
    'Team_Succ_onxGA': 'Goles esperados en contra del equipo',
    'Team_Succ_xG+/-': 'Diferencia de goles esperados del equipo',
    'Team_Succ_xG+/-90': 'Diferencia de goles esperados por 90 minutos',
    'Team_Succ_On-Off.1': 'Eficacia del equipo con/sin jugador (variante)',
    '2CrdY': 'Doble amarilla',
    'Fls': 'Faltas cometidas',
    'Fld': 'Faltas recibidas (FBref)',
    'Off': 'Fuera de juego',
    'Crs': 'Pases cruzados',
    'PKwon': 'Penales ganados',
    'PKcon': 'Penales concedidos',
    'OG': 'Autogoles',
    'Recov': 'Recuperaciones',
    'Aerial_Won': 'Duelos aéreos ganados',
    'Aerial_Lost': 'Duelos aéreos perdidos',
    'Aerial_Won%': 'Porcentaje de duelos aéreos ganados (FBref)',
    'Gen. Role': 'Rol general',
    'Role': 'Rol específico',
    'xGoalsAdded': 'Goles añadidos esperados',
    'xGoalsAdded_p90': 'Goles añadidos esperados por 90 minutos',
    'DAVIES': 'DAVIES',
    'DAVIES_Box Activity': 'Actividad en el área (DAVIES)',
    'DAVIES_Shooting': 'Disparos (DAVIES)',
    'DAVIES_Final Ball': 'Último pase (DAVIES)',
    'DAVIES_Dribbles and Carries': 'Regates y conducciones (DAVIES)',
    'DAVIES_Buildup Passing': 'Construcción de juego (DAVIES)',
    'DAVIES_Defense': 'Defensa (DAVIES)',
    'DAVIES_p90': 'DAVIES por 90 minutos',
    'DAVIES_Box Activity_p90': 'Actividad en el área por 90 minutos (DAVIES)',
    'DAVIES_Shooting_p90': 'Disparos por 90 minutos (DAVIES)',
    'DAVIES_Final Ball_p90': 'Último pase por 90 minutos (DAVIES)',
    'DAVIES_Dribbles and Carries_p90': 'Regates y conducciones por 90 minutos (DAVIES)',
    'DAVIES_Buildup Passing_p90': 'Construcción de juego por 90 minutos (DAVIES)',
    'DAVIES_Defense_p90': 'Defensa por 90 minutos (DAVIES)',
    'team_elo': 'ELO del equipo',
    'team_rank': 'Ranking del equipo'
}



# CSS para estilizar la tabla
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


# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df[df.season!='2023-2024']
    return df

def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

df = load_data()

# Inverso del diccionario de mapeo para buscar la clave original por el valor mapeado
estadisticas_inverso = {v: k for k, v in estadisticas.items()}
posiciones_inverso = {v: k for k, v in posiciones.items()}
ligas_inverso = {v: k for k, v in ligas.items()}

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
player_position = posiciones.get(player_position, player_position)

player_age = player['age'].values[0]
player_nation = player['nation'].values[0]
player_starts = player['Starts'].values[0]

# HTML para mostrar los datos del jugador
html = f"""
{css}
<div class="bio-grid">
    <div class="bio-stat">
        <div class="bio-stat-heading">Nombre</div>
        <div class="bio-stat-value" id="player-name"><span class="white-text">{player_name}</span></div>
    </div>
    <div class="bio-stat">
        <div class="bio-stat-heading">Posición</div>
        <div class="bio-stat-value" id="player-position"><span class="white-text">{player_position}</span></div>
    </div>
    <div class="bio-stat">
        <div class="bio-stat-heading">Edad</div>
        <div class="bio-stat-value" id="player-age"><span class="white-text">{player_age}</span></div>
    </div>
    <div class="bio-stat">
        <div class="bio-stat-heading">Nacionalidad</div>
        <div class="bio-stat-value" id="player-nation"><span class="white-text">{player_nation}</span></div>
    </div>
    <div class="bio-stat">
        <div class="bio-stat-heading">Partidos titular</div>
        <div class="bio-stat-value" id="player-starts"><span class="white-text">{player_starts}</span></div>
    </div>
</div>
"""

# Mostrar la tabla de biografía del jugador en Streamlit
st.markdown(html, unsafe_allow_html=True)
estadisticas_inverso = {v: k for k, v in estadisticas.items()}
player_position = posiciones_inverso.get(player_position, player_position)

df_columns_mapeadas = [estadisticas.get(col, col) for col in df.select_dtypes(include=np.number).columns]
selected_features_mapeadas = st.multiselect('Seleccionar características clave:', df_columns_mapeadas)
selected_features = [estadisticas_inverso.get(feature, feature) for feature in selected_features_mapeadas]

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
        
        feature = estadisticas.get(feature, feature)

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