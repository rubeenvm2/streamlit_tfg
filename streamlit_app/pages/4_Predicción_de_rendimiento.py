import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt
import pickle
columnas = [
    'groundDuelsWonPercentage', 'aerialDuelsWonPercentage', 'wasFouled', 'dispossessed', 
    'accurateFinalThirdPasses', 'bigChancesCreated', 'keyPasses', 'accurateCrossesPercentage', 
    'accurateLongBallsPercentage', 'dribbledPast', 'bigChancesMissed', 'hitWoodwork', 
    'errorLeadToGoal', 'errorLeadToShot', 'passToAssist', 'pos', 'age', 'born', 'MP', 
    'Starts', '90s', 'PK', 'PKatt', 'CrdY', 'CrdR', 'PrgC', 'PrgP', 'PrgR', 'Gls_90', 
    'Ast_90', 'G+A_90', 'G-PK_90', 'G+A-PK_90', 'xG_90', 'xAG_90', 'xG+xAG_90', 'npxG_90', 
    'npxG+xAG_90', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Dist', 'FK', 'npxG/Sh', 
    'G-xG', 'np:G-xG', 'Total_Att', 'Total_Cmp%', 'Total_TotDist', 'Total_PrgDist', 
    'Short_Att', 'Short_Cmp%', 'Medium_Att', 'Medium_Cmp%', 'Long_Att', 'Long_Cmp%', 
    'xA', 'A-xAG', 'KP', '1/3', 'PPA', 'CrsPA', 'Att', 'Pass Types_Live', 'Pass Types_Dead', 
    'Pass Types_FK', 'Pass Types_TB', 'Pass Types_Sw', 'Pass Types_Crs', 'Pass Types_TI', 
    'Pass Types_CK', 'Corner Kicks_In', 'Corner Kicks_Out', 'Corner Kicks_Str', 'Outcomes_Cmp', 
    'Outcomes_Off', 'Outcomes_Blocks', 'SCA90', 'SCA_PassLive', 'SCA_PassDead', 'SCA_TO', 
    'SCA_Sh', 'SCA_Fld', 'SCA_Def', 'GCA90', 'GCA_PassLive', 'GCA_PassDead', 'GCA_TO', 
    'GCA_Sh', 'GCA_Fld', 'GCA_Def', 'Tkl', 'TklW', 'Tackles_Def 3rd', 'Tackles_Mid 3rd', 
    'Tackles_Att 3rd', 'Chall_Att', 'Chall_Tkl%', 'Chall_Lost', 'Blocks', 'Blocks_Sh', 
    'Blocks_Pass', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches', 'Touches_Def Pen', 
    'Touches_Def 3rd', 'Touches_Mid 3rd', 'Touches_Att 3rd', 'Touches_Att Pen', 'Touches_Live', 
    'Take-Ons_Att', 'Take-Ons_Succ%', 'Take-Ons_Tkld%', 'Carries', 'Carries_TotDist', 
    'Carries_PrgDist', 'Carries_PrgC', 'Carries_1/3', 'Carries_CPA', 'Carries_Mis', 
    'Carries_Dis', 'Receiving_Rec', 'Receiving_PrgR', 'Mn/MP', 'Min%', 'Compl', 'Subs', 
    'unSub', 'Team_Succ_PPM', 'Team_Succ_onG', 'Team_Succ_onGA', 'Team_Succ_+/-90', 
    'Team_Succ_On-Off', 'Team_Succ_onxG', 'Team_Succ_onxGA', 'Team_Succ_xG+/-90', 
    'Team_Succ_On-Off.1', '2CrdY', 'Fls', 'Fld', 'Off', 'Crs', 'PKwon', 'PKcon', 'OG', 
    'Recov', 'Aerial_Lost', 'Aerial_Won%', 'Gen. Role', 'Role', 'xGoalsAdded_p90', 'DAVIES', 
    'team_elo', 'team_rank', 'DAVIES_next_season'
]
st.set_page_config(
page_title="Predicción de rendimiento",
page_icon="📈",
layout="wide",
initial_sidebar_state="expanded")
# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df_no_dav = pd.read_csv('streamlit_app/model_output_no_dav.csv')
    df_dav = pd.read_csv('streamlit_app/model_output_dav.csv')
    with open('streamlit_app/lightgbm_30cols_davies.pkl', 'rb') as file:
        data = pickle.load(file)
    return df,df_no_dav,df_dav, data

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

def map_positions(positions):
    split = positions.split(",")
    if len(split) > 1:
        positions = split[0]
    return positions

st.title("Predicción de rendimiento")

df, df_no_dav, df_dav, data = load_data()
try:
    similar_player = st.session_state.similar_player
    similar_team = st.session_state.similar_team
    if similar_team:
        categorical_cols = []
        df_dav['pos'] = df_dav['pos'].apply(map_positions)
        df_no_dav['pos'] = df_no_dav['pos'].apply(map_positions)
        df['pos'] = df['pos'].apply(map_positions)
        df['DAVIES_next_season'] = 0
        
        for column in df_dav.select_dtypes(include=['object']).columns:
            if column not in ['player', 'team', 'league', 'season', 'nation']:
                categorical_cols.append(column)
                df_dav[column] = data[f'le_{column}'].transform(df_dav[column])
                df_no_dav[column] = data[f'le_{column}'].transform(df_no_dav[column])
                df[column] = data[f'le_{column}'].transform(df[column])
        
        df_dav[columnas] = data["Scaler"].inverse_transform(df_dav[columnas])
        df[columnas] = data["Scaler"].inverse_transform(df[columnas])
        df_no_dav[columnas] = data["Scaler"].inverse_transform(df_no_dav[columnas])
        
        df_dav.loc[df_dav['pos'] == 4, 'pos'] = 2
        df_no_dav.loc[df_no_dav['pos'] == 4, 'pos'] = 2
        df_dav.pos = df_dav.pos.astype('int32')
        
        df_no_dav.loc[df_no_dav['pos'] == 2, 'pos'] = 1
        df_dav.loc[df_dav['pos'] == 2, 'pos'] = 1
        df_no_dav.pos = df_no_dav.pos.astype('int32')
        
        df_dav.loc[df_dav['Gen. Role'] == 12, 'Gen. Role'] = 2
        df_no_dav.loc[df_no_dav['Gen. Role'] == 12, 'Gen. Role'] = 2
        
        df_no_dav.loc[df_no_dav['Gen. Role'] == 36, 'Gen. Role'] = 6
        df_dav.loc[df_dav['Gen. Role'] == 36, 'Gen. Role'] = 6
        
        df_dav.loc[df_dav['Gen. Role'] == 30, 'Gen. Role'] = 5
        df_no_dav.loc[df_no_dav['Gen. Role'] == 30, 'Gen. Role'] = 5
        
        df_dav.loc[df_dav['Gen. Role'] == 6, 'Gen. Role'] = 1
        df_no_dav.loc[df_no_dav['Gen. Role'] == 6, 'Gen. Role'] = 1
        
        df_dav.loc[df_dav['Gen. Role'] == 18, 'Gen. Role'] = 3
        df_no_dav.loc[df_no_dav['Gen. Role'] == 18, 'Gen. Role'] = 3
        
        df_dav.loc[df_dav['Gen. Role'] == 24, 'Gen. Role'] = 4
        df_no_dav.loc[df_no_dav['Gen. Role'] == 24, 'Gen. Role'] = 4
        df_dav['Gen. Role'] = df_dav['Gen. Role'].astype('int32')
        df_no_dav['Gen. Role'] = df_no_dav['Gen. Role'].astype('int32')
        
        df_dav['Role'] = df_dav['Role'].rank(method='dense') - 1
        df_no_dav['Role'] = df_no_dav['Role'].rank(method='dense') - 1
        df_dav['Role'] = df_dav['Role'].astype('int32')
        df_no_dav['Role'] = df_no_dav['Role'].astype('int32')
        
        for column in categorical_cols:
            print(column)
            df_dav[column] = data[f'le_{column}'].inverse_transform(df_dav[column])
            df_no_dav[column] = data[f'le_{column}'].inverse_transform(df_no_dav[column])
        
        
        
        st.write(f"En esta página primeramente verás una estadística de la media de rendimiento para cada temporada de los jugadores existentes en la base de datos. Estas puntuaciones van de 0 a infinito y han sido predichas mediante un modelo de machine learning basandose en todo tipo de estadisticas, tanto defensivas como ofensivas, como de creación de juego, progresión con el balón. Captando así el estilo de juego de {similar_player} para predecir correctamente el rendimiento.")
        create_scatterplots(df, similar_player)
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

except:
    st.write("Selecciona en la pestaña de análisis de similitud a un jugador y su equipo para comparar antes de hacer la predicción del rendimiento.")
