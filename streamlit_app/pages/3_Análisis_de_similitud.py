import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
page_title="Análisis de similitud",
page_icon="📝",
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


# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season!='2023-2024']
    df_no_dav = pd.read_csv('streamlit_app/model_output_no_dav.csv')
    df_dav = pd.read_csv('streamlit_app/model_output_dav.csv')
    return df,df_no_dav,df_dav

# Función para normalizar las características seleccionadas
def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# Función para encontrar jugadores similares
def find_similar_players(df, target_player, target_team, features, top_n=10, leagues=None, teams=None, age_range=None):
    # Filtrar solo las características numéricas
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    df_normalized = normalize_features(df.copy(), numeric_features)
    target_position = df[(df['player'] == target_player.player) & (df['team'] == target_team) & (df['season'] == '2022-2023')]['pos'].values[0].split(",")
    target_position = [pos.strip() for pos in target_position]  # Eliminar espacios adicionales

    df = df[df['pos'].apply(lambda x: any(pos in x.split(",") for pos in target_position))]
    target_vector = df_normalized[(df_normalized['player'] == target_player.player) & (df_normalized['team'] == target_team) & (df['season'] == '2022-2023')][numeric_features].values
    if leagues:
        df = df[df['league'].isin(leagues)]
    if teams:
        df = df[df['team'].isin(teams)]
    if age_range:
        df = df[df['age'] <= age_range]

    # Calcular la similitud de coseno
    similarity_matrix = cosine_similarity(df_normalized[numeric_features], target_vector)
    # Añadir la similitud al dataframe y convertir a porcentaje
    df_normalized['similarity'] = similarity_matrix[:, 0] * 100
    df['similarity'] = df_normalized.loc[df.index].similarity

    # Filtrar jugadores según los filtros opcionales
    similar_players = df[(df['season'] == '2022-2023') & (df['player'] != target_player.player)]
    
    similar_players = similar_players.sort_values(by='similarity', ascending=False).head(top_n)
    
    return similar_players

# Función para crear un radar plot
def create_radar_plot(selected_player, player_data, features):
                        
    # Crea el objeto Radar
    radar = Radar(background_color="#121212", patch_color="#28252C", label_color="#FFFFFF", range_color="#FFFFFF")

    ## title
    title = dict(
        title_name=selected_player.player.unique()[0],
        title_color='#9B3647',
        subtitle_name=selected_player.team.unique()[0],
        subtitle_color='#ABCDEF',
        title_name_2=player_data.player.unique()[0],
        title_color_2='#3282b8',
        subtitle_name_2=player_data.team.unique()[0],
        subtitle_color_2='#ABCDEF',
        title_fontsize=18,
        subtitle_fontsize=15,
    )
    # Crea el gráfico de radar
    fig, ax = radar.plot_radar(ranges=[(0, 1) for feature in features], 
                            params= [estadisticas.get(col, col) for col in features], 
                            values=[selected_player[features].values.tolist()[0],player_data[features].values.tolist()[0]], 
                            radar_color=['#B6282F', '#344D94'], 
                            title=title, alphas =[0.55, 0.5],
                            compare=True)
    st.pyplot(fig)
    
# Función para crear la historia del jugador
def create_player_history(df, selected_player_name, player_name, selected_features):
    # Filtrar el DataFrame para obtener el historial del jugador especificado
    player_history = df[df['player'] == player_name]
    selected_player_history = df[df['player']==selected_player_name]
    relevant_seasons = player_history['season'].unique()
    selected_player_history = selected_player_history[selected_player_history.season.isin(relevant_seasons)]
    player_history = player_history[player_history.season.isin(relevant_seasons)]
    df = df[df.season.isin(relevant_seasons)]

    if player_history.empty:
        st.write(f"No se encontró historial para el jugador {player_name}.")
        return

    # Obtener la posición del jugador
    player_position = player_history['pos'].iloc[0]

    for i, feature in enumerate(selected_features):
        if feature in df.columns:
            feature_mapped = estadisticas.get(feature, feature)
            fig = go.Figure()
            # Filtrar las filas del jugador que tienen un valor para la estadística actual
            player_feature_history = player_history
            
            if not player_feature_history.empty:
                # Añadir puntos del jugador
                player_feature_history = player_feature_history.sort_values(by='season')
                
                fig.add_trace(go.Scatter(
                    x=player_feature_history['season'],
                    y=player_feature_history[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in player_feature_history[feature]],
                    textposition='top center',
                    name=f'{player_name} - {feature_mapped}',
                    legendgroup=f'{player_name} - {feature_mapped}'
                ))

                # Calcular y añadir las medianas por temporada y posición
                medianas = df[(df['pos'] == player_position)].groupby('season')[feature].median().reset_index()

                # Añadir la línea de medias por posición
                fig.add_trace(go.Scatter(
                    x=medianas['season'],
                    y=medianas[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in medianas[feature]],
                    textposition='top center',
                    name=f'Mediana de su posición - {feature_mapped}',
                    legendgroup=f'Mediana de su posición - {feature_mapped}',
                    line=dict(color='red'),
                    marker=dict(color='red', size=10)
                ))
                
                fig.add_trace(go.Scatter(
                    x=selected_player_history['season'],
                    y=selected_player_history[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in selected_player_history[feature]],
                    textposition='top center',
                    name=f'{selected_player_name} - {feature_mapped}',
                    legendgroup=f'{selected_player_name} - {feature_mapped}',
                    line=dict(color='green'),
                    marker=dict(color='green', size=10)
                ))

            else:
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[0],
                    mode='text',
                    text=f"No se encontró historial para el jugador {player_name} en la característica '{feature_mapped}'.",
                    showlegend=False
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='text',
                text=f"La característica '{feature_mapped}' no se encuentra en el DataFrame.",
                showlegend=False
            ))

        # Configurar las etiquetas y el título del gráfico
        fig.update_layout(
            title=f'Comparación histórica de {feature_mapped}',
            xaxis_title='Season',
            yaxis_title='Valor',
            legend_title='Leyenda',
            legend=dict(
                x=1,  # Posición horizontal (0: izquierda, 1: derecha)
                y=1,  # Posición vertical (0: abajo, 1: arriba)
                orientation='v'  # Orientación de la leyenda (opciones: 'v', 'h')
            )
        )

        st.plotly_chart(fig)


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

def select_features(df_columns_mapeadas):
    if 'selected_features' not in st.session_state:
        # Si no hay características seleccionadas en la sesión, establecerlas en una lista vacía
        st.session_state['selected_features'] = []


    # Selección de características numéricas
    if stateful_button('Selecciona todas las características.',sidebar=True, key="select_features"):
        st.session_state.selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    default_cols = [estadisticas.get(feature, feature) for feature in st.session_state.selected_features]
    st.session_state.selected_features = st.sidebar.multiselect('Selecciona las características', df_columns_mapeadas, default=default_cols)
    selected_features = st.session_state.selected_features

    return selected_features

estadisticas_inverso = {v: k for k, v in estadisticas.items()}
posiciones_inverso = {v: k for k, v in posiciones.items()}
ligas_inverso = {v: k for k, v in ligas.items()}

# Cargar datos
df, df_no_dav, df_dav = load_data()
target_player = st.session_state.player
# Título de la aplicación
st.title('Búsqueda de jugadores similares')
st.write(f"En esta página puedes observar aquellos jugadores con un perfil similar a {target_player.player} de tal manera que puedas encontrar al sustituto ideal.")
# Columnas para los filtros principales
selected_player = st.session_state.player
selected_team = st.session_state.player.team

# Selección de características y filtros adicionales
st.header('Filtros Adicionales')
col3, col4 = st.columns(2)
with col3:
    filter_by_league = st.checkbox('Filtrar por ligas')
    if filter_by_league:
        df_ligas_mapeadas = [ligas.get(liga, liga) for liga in df.league.unique()]
        selected_leagues_mapeadas = st.multiselect('Selecciona las ligas', df_ligas_mapeadas)
        selected_leagues = [ligas_inverso.get(liga, liga) for liga in selected_leagues_mapeadas]
        filtered_df = df[df['league'].isin(selected_leagues)]
    else:
        selected_leagues = None
    
    filter_by_team = st.checkbox('Filtrar por equipos')
    if filter_by_team:
        selected_teams = st.multiselect('Selecciona los equipos', filtered_df['team'].unique())
        filtered_df = df[df['team'].isin(selected_leagues)]
    else:
        selected_teams = None
with col4:
    filter_by_age = st.checkbox('Filtrar por rango de edad')
    if filter_by_age:
        age_range = st.slider('Selecciona el rango de edad', min_value=int(filtered_df['age'].min()), max_value=int(filtered_df['age'].max()))
        filtered_df = df[df['age'] <= age_range]
    else:
        age_range = None
df_columns_mapeadas = [estadisticas.get(col, col) for col in df.select_dtypes(include=[np.number]).columns.tolist()]
selected_features = select_features(df_columns_mapeadas)

# Mostrar solo algunas características seleccionadas y un contador
MAX_DISPLAYED_FEATURES = 5
if len(selected_features) <= MAX_DISPLAYED_FEATURES:
    selected_features_display = ", ".join(selected_features)
else:
    selected_features_display = ", ".join(selected_features[:MAX_DISPLAYED_FEATURES]) + f", y {len(selected_features) - MAX_DISPLAYED_FEATURES} más"

st.write(f"Características seleccionadas: {selected_features_display}")
selected_features = [estadisticas_inverso.get(feature, feature) for feature in st.session_state.selected_features]

# Slider para seleccionar el número de jugadores similares a mostrar
top_n = st.slider('Número de jugadores similares a mostrar', 1, 20, 10)

# Botón para ejecutar el análisis
if selected_team and len(selected_features) > 0:
    similar_players = find_similar_players(df, selected_player, selected_team, selected_features, top_n, selected_leagues, selected_teams, age_range)
    st.header(f'Jugadores Similares a {selected_player.player} en {selected_team}')
    selected_player_data = df[(df['player'] == selected_player.player) & (df['season'] == '2022-2023') & (df.team == selected_team)]
    for i, row in similar_players.iterrows():
        st.write(f"Jugador: {row['player']} | Equipo: {row['team']} | Similitud: {row['similarity']:.2f}%")

    # Mostrar lista visual de jugadores similares con el porcentaje de similitud
    similar_players_list = []
    for i, row in similar_players.iterrows():
        similar_players_list.append(f"{row['player']} | {row['team']} | Age: {row['age']} | Similitud: {row['similarity']:.2f}%")

    selected_similar_player = st.selectbox('Selecciona un jugador para ver una comparación más exhaustiva.', similar_players_list)
    if selected_similar_player != None:
        # Botón para mostrar la tabla completa
        similar_players = pd.concat([selected_player_data, similar_players])
        if stateful_button('Mostrar tabla con las estadísticas.', key="similar_players"):
        #if st.button('Mostrar detalles completos'):
            print(selected_features)
            st.write(similar_players[['player', 'team', 'pos'] + ['similarity'] + selected_features ])
        
        # Crear radar plot para el jugador seleccionado
        similar_player_name = selected_similar_player.split('|')[0].strip()
        df_normalized = normalize_features(df.copy(), selected_features)
        if len(selected_features) < 15 and len(selected_features) >= 3:
            similar_player_data_norm = df_normalized[(df['player'] == similar_player_name) & (df['season'] == '2022-2023')]
            selected_player_data_norm = df_normalized[(df['player'] == selected_player.player) & (df['season'] == '2022-2023') & (df.team == selected_team)]

            create_radar_plot(selected_player_data_norm, similar_player_data_norm, selected_features)
                # Crear historia del jugador seleccionado
            st.header("Histórico del jugador seleccionado.")
            create_player_history(df, selected_player.player, similar_player_name, selected_features)
        else:
            st.write("Por favor, selecciona menos de 15 o más de 3 características para poder ver un radar plot y el histórico del jugador escogido en las diferentes características")
        st.session_state.similar_player = similar_player_name
        st.session_state.similar_team = selected_similar_player.split('|')[1].strip()
        st.write(f"Presionando el botón en la parte inferior serás redirigido a una página para poder tener una predicción de cual será el rendimiento de {similar_player_name} en la temporada siguiente.")
        st.page_link("pages/4_Predicción_de_rendimiento.py", label="Botón")
    else:
        st.write(f"No existe jugador suficientemente parecido a {target_player.player}. Selecciona otra combinación de características o sé menos exhaustivo con los filtros.")
else:
    st.write(f"Porfavor, selecciona almenos 3 características para poder encontrar el jugador que más se asemeje a {target_player.player} basandose en las estadísticas seleccionadas.")
    st.write("También puedes clicar el botón situado en la barra lateral para seleccionar todas las características.")
