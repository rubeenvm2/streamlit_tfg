import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go

columnas = [
'groundDuelsWonPercentage',
'aerialDuelsWonPercentage',
'wasFouled',
'dispossessed',
'accurateFinalThirdPasses',
'bigChancesCreated',
'keyPasses',
'accurateCrossesPercentage',
'accurateLongBallsPercentage',
'dribbledPast',
'bigChancesMissed',
'hitWoodwork',
'errorLeadToGoal',
'errorLeadToShot',
'passToAssist',
'pos',
'age',
'born',
'MP',
'Starts',
'90s',
'PK',
'PKatt',
'CrdY',
'CrdR',
'PrgC',
'PrgP',
'PrgR',
'Gls_90',
'Ast_90',
'G+A_90',
'G-PK_90',
'G+A-PK_90',
'xG_90',
'xAG_90',
'xG+xAG_90',
'npxG_90',
'npxG+xAG_90',
'SoT%',
'Sh/90',
'SoT/90',
'G/Sh',
'G/SoT',
'Dist',
'FK',
'npxG/Sh',
'G-xG',
'np:G-xG',
'Total_Att',
'Total_Cmp%',
'Total_TotDist',
'Total_PrgDist',
'Short_Att',
'Short_Cmp%',
'Medium_Att',
'Medium_Cmp%',
'Long_Att',
'Long_Cmp%',
'xA',
'A-xAG',
'KP',
'1/3',
'PPA',
'CrsPA',
'Att',
'Pass Types_Live',
'Pass Types_Dead',
'Pass Types_FK',
'Pass Types_TB',
'Pass Types_Sw',
'Pass Types_Crs',
'Pass Types_TI',
'Pass Types_CK',
'Corner Kicks_In',
'Corner Kicks_Out',
'Corner Kicks_Str',
'Outcomes_Cmp',
'Outcomes_Off',
'Outcomes_Blocks',
'SCA90',
'SCA_PassLive',
'SCA_PassDead',
'SCA_TO',
'SCA_Sh',
'SCA_Fld',
'SCA_Def',
'GCA90',
'GCA_PassLive',
'GCA_PassDead',
'GCA_TO',
'GCA_Sh',
'GCA_Fld',
'GCA_Def',
'Tkl',
'TklW',
'Tackles_Def 3rd',
'Tackles_Mid 3rd',
'Tackles_Att 3rd',
'Chall_Att',
'Chall_Tkl%',
'Chall_Lost',
'Blocks',
'Blocks_Sh',
'Blocks_Pass',
'Int',
'Tkl+Int',
'Clr',
'Err',
'Touches',
'Touches_Def Pen',
'Touches_Def 3rd',
'Touches_Mid 3rd',
'Touches_Att 3rd',
'Touches_Att Pen',
'Touches_Live',
'Take-Ons_Att',
'Take-Ons_Succ%',
'Take-Ons_Tkld%',
'Carries',
'Carries_TotDist',
'Carries_PrgDist',
'Carries_PrgC',
'Carries_1/3',
'Carries_CPA',
'Carries_Mis',
'Carries_Dis',
'Receiving_Rec',
'Receiving_PrgR',
'Mn/MP',
'Min%',
'Compl',
'Subs',
'unSub',
'Team_Succ_PPM',
'Team_Succ_onG',
'Team_Succ_onGA',
'Team_Succ_+/-90',
'Team_Succ_On-Off',
'Team_Succ_onxG',
'Team_Succ_onxGA',
'Team_Succ_xG+/-90',
'Team_Succ_On-Off.1',
'2CrdY',
'Fls',
'Fld',
'Off',
'Crs',
'PKwon',
'PKcon',
'OG',
'Recov',
'Aerial_Lost',
'Aerial_Won%',
'Gen. Role',
'Role',
'xGoalsAdded_p90',
'team_elo',
'team_rank',
'DAVIES'
]

st.set_page_config(
page_title="Predicción de rendimiento",
page_icon="📈",
layout="wide",
initial_sidebar_state="expanded")
# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df[df.season!='2023-2024']
    with open('lightgbm_30cols_davies.pkl', 'rb') as file:
        data = pickle.load(file)
    with open('xgboost_30cols_no_davies.pkl', 'rb') as file:
        data2 = pickle.load(file)
    return df, data, data2

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

def create_linechart(df, player):
    all_seasons = sorted(df['season'].unique(), key=lambda x: int(x.split('-')[0]))
    player_data = df[(df.player == player) & (df.season != '2023-2024')]
    df = df[df.pos == player_data.pos.unique()[0]]
    # Crear un scatter plot interactivo para los datos del jugador
    fig = go.Figure()
    
    # Añadir puntos del jugador
    fig.add_trace(go.Scatter(
        x=player_data['season'],
        y=player_data['DAVIES'],
        mode='lines+markers+text',
        text=[f'{val:.2f}' for val in player_data['DAVIES']],
        textposition='top center',
        marker=dict(color='blue', size=10),
        name=player
    ))

    # Calcular y añadir las medianas por temporada y posición
    relevant_seasons = player_data['season'].unique()
    medianas = df[df['season'].isin(relevant_seasons)].groupby('season')['DAVIES'].mean().reset_index()
    key_func = lambda x: int(x.split('-')[0])

    # Aplica la función lambda a la columna 'season' para extraer el primer año de cada temporada
    medianas['season_first_year'] = medianas['season'].apply(key_func)

    # Ordena las temporadas utilizando el primer año como clave
    medianas_sorted = medianas.sort_values(by='season_first_year')

    # Asigna las temporadas ordenadas de vuelta a la columna 'season' en medianas
    medianas['season'] = pd.Categorical(medianas_sorted['season'], categories=all_seasons, ordered=True)

    # Elimina la columna temporal 'season_first_year' si ya no es necesaria
    medianas.drop(columns=['season_first_year'], inplace=True)

    # Añadir la línea de medias por posición
    fig.add_trace(go.Scatter(
        x=medianas['season'],
        y=medianas['DAVIES'],
        mode='lines+markers+text',
        text=[f'{val:.2f}' for val in medianas['DAVIES']],
        textposition='top center',
        line=dict(color='red'),
        name='Media de su posición'
    ))

    # Configurar las etiquetas y el título del gráfico
    fig.update_layout(
        title=f'Histórico de {player} - DAVIES',
        xaxis_title='Season',
        yaxis_title='DAVIES',
        legend_title='Leyenda',
        legend=dict(x=0, y=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig)

def map_positions(positions):
    split = positions.split(",")
    if len(split) > 1:
        positions = split[0]
    return positions

def compute_shap_values(model, data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values

def plot_shap_waterfall(shap_values, input_data):
    fig, ax = plt.subplots()

    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)
st.title("Predicción de rendimiento")

df, data, data2 = load_data()

model = data['model']
model2 = data2['model']
similar_player = st.session_state.similar_player
similar_team = st.session_state.similar_team

st.write(f"En esta página primeramente verás una estadística de la media de rendimiento para cada temporada de los jugadores existentes en la base de datos. Estas puntuaciones van de 0 a infinito y han sido predichas mediante un modelo de machine learning basandose en todo tipo de estadisticas, tanto defensivas como ofensivas, como de creación de juego, progresión con el balón. Captando así el estilo de juego de {similar_player} para predecir correctamente el rendimiento.")
create_linechart(df, similar_player)
player_df = df.copy()
player_df = player_df[player_df.season=='2022-2023']
player_df = player_df[player_df['player'] == similar_player]
player_df['pos'] = player_df['pos'].apply(map_positions)
player_df['DAVIES_next_season'] = 0

categorical_cols = player_df.select_dtypes(include=object).columns 
for column in categorical_cols:
    if column not in ['nation', 'league', 'player', 'season', 'team']:
        player_df[column] = data[f"le_{column}"].transform(player_df[column])

# Access each item in the dictionary
model = data["model"]
X_test = data["x_test"]
y_test = data["y_test"]
scaler = data["Scaler"]

model2 = data2["model"]
X_test2 = data2["x_test"]
y_test2 = data2["y_test"]
scaler2 = data2["Scaler"]

if similar_team:
    #actual_value = df_dav[(df_dav.player == similar_player) & (df_dav.season == '2022-2023') & (df_dav.team == similar_team)]['DAVIES'].unique()[0]
    col1,col2 = st.columns(2)
    with col1:
        if stateful_button(f'Predecir rendimiento de {similar_player} en la temporada siguiente a la actual. (con DAVIES)', key="pred_DAV"):
            actual_value = player_df.DAVIES.unique()[0]
            player_df = player_df[columnas]
            cols = [col for col in player_df.columns if col != 'DAVIES_next_season' and col != 'DAVIES']
            player_davies = player_df.copy()
            player_davies[cols] = scaler.transform(player_df[cols])
            X = player_davies[X_test.columns]
            prediction_value = model.predict(X)[0]
            st.write(f"La performance de este año del jugador ha sido de {actual_value:.2f}. La predicción para el año siguiente es {prediction_value:.2f}")
            st.write('Gráfico de Waterfall de SHAP Values:')
            shap_values = compute_shap_values(model, X)
            plot_shap_waterfall(shap_values, X)

    with col2:
        if stateful_button(f'Predecir rendimiento de {similar_player} en la temporada siguiente a la actual. (sin DAVIES)', key="pred_no_DAV"):
            actual_value = player_df.DAVIES.unique()[0]
            player_df = player_df[columnas]
            player_no_davies = player_df.copy()
            player_no_davies = player_no_davies.drop('DAVIES', axis=1)
            cols = [col for col in player_no_davies.columns if col != 'DAVIES_next_season' and col != 'DAVIES']
            player_no_davies[cols] = scaler2.transform(player_no_davies[cols])
            X = player_no_davies[X_test2.columns]
            shap_values = compute_shap_values(model2, X)
            prediction_value = shap_values[0].base_values + shap_values[0].values.sum()
            st.write(f"La performance de este año del jugador ha sido de {actual_value:.2f}. La predicción para el año siguiente es {prediction_value:.2f}")
            st.write('Gráfico de Waterfall de SHAP Values:')
            plot_shap_waterfall(shap_values, X)