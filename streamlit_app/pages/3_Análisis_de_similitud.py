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

# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df[df.season!='2023-2024']
    df_no_dav = pd.read_csv('model_output_no_dav.csv')
    df_dav = pd.read_csv('model_output_dav.csv')
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
        df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
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
                            params=features, 
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
            fig = go.Figure()
            # Filtrar las filas del jugador que tienen un valor para la estadística actual
            player_feature_history = player_history

            if not player_feature_history.empty:
                # Añadir puntos del jugador

                
                fig.add_trace(go.Scatter(
                    x=player_feature_history['season'],
                    y=player_feature_history[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in player_feature_history[feature]],
                    textposition='top center',
                    name=f'{player_name} - {feature}',
                    legendgroup=f'{player_name} - {feature}'
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
                    name=f'Mediana de su posición - {feature}',
                    legendgroup=f'Mediana de su posición - {feature}',
                    line=dict(color='red'),
                    marker=dict(color='red', size=10)
                ))
                
                fig.add_trace(go.Scatter(
                    x=selected_player_history['season'],
                    y=selected_player_history[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in selected_player_history[feature]],
                    textposition='top center',
                    name=f'{selected_player_name} - {feature}',
                    legendgroup=f'{selected_player_name} - {feature}',
                    line=dict(color='green'),
                    marker=dict(color='green', size=10)
                ))

            else:
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[0],
                    mode='text',
                    text=f"No se encontró historial para el jugador {player_name} en la característica '{feature}'.",
                    showlegend=False
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='text',
                text=f"La característica '{feature}' no se encuentra en el DataFrame.",
                showlegend=False
            ))

        # Configurar las etiquetas y el título del gráfico
        fig.update_layout(
            title=f'Comparación histórica de {feature}',
            xaxis_title='Season',
            yaxis_title='Valor',
            legend_title='Leyenda',
            legend=dict(
                x=1,  # Posición horizontal (0: izquierda, 1: derecha)
                y=1,  # Posición vertical (0: abajo, 1: arriba)
                orientation='h'  # Orientación de la leyenda (opciones: 'v', 'h')
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

def select_features():
    if 'selected_features' not in st.session_state:
        # Si no hay características seleccionadas en la sesión, establecerlas en una lista vacía
        st.session_state['selected_features'] = []

    # Selección de características numéricas
    if stateful_button('Selecciona todas las características.',sidebar=True, key="select_features"):
    #if st.sidebar.button("Selecciona todas las características."):
        st.session_state.selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.selected_features = st.sidebar.multiselect('Selecciona las características', df.select_dtypes(include=[np.number]).columns.tolist(), default=st.session_state.selected_features)
    selected_features = st.session_state.selected_features
    
    return selected_features

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
        selected_leagues = st.multiselect('Selecciona las ligas', df['league'].unique())
    else:
        selected_leagues = None
    
    filter_by_team = st.checkbox('Filtrar por equipos')
    if filter_by_team:
        selected_teams = st.multiselect('Selecciona los equipos', df['team'].unique())
    else:
        selected_teams = None
with col4:
    filter_by_age = st.checkbox('Filtrar por rango de edad')
    if filter_by_age:
        age_range = st.slider('Selecciona el rango de edad', min_value=int(df['age'].min()), max_value=int(df['age'].max()))
    else:
        age_range = None

selected_features = select_features()

# Mostrar solo algunas características seleccionadas y un contador
MAX_DISPLAYED_FEATURES = 5
if len(selected_features) <= MAX_DISPLAYED_FEATURES:
    selected_features_display = ", ".join(selected_features)
else:
    selected_features_display = ", ".join(selected_features[:MAX_DISPLAYED_FEATURES]) + f", y {len(selected_features) - MAX_DISPLAYED_FEATURES} más"

st.write(f"Características seleccionadas: {selected_features_display}")
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
    
    # Botón para mostrar la tabla completa
    similar_players = pd.concat([selected_player_data, similar_players])
    if stateful_button('Mostrar tabla con las estadísticas.', key="similar_players"):
    #if st.button('Mostrar detalles completos'):
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
    st.write(f"Porfavor, selecciona almenos 3 características para poder encontrar el jugador que más se asemeje a {target_player.player} basandose en las estadísticas seleccionadas.")
    st.write("También puedes clicar el botón situado en la barra lateral para seleccionar todas las características.")