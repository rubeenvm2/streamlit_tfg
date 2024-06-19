import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
page_title="Identificación posición débil",
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
    'Att': 'Pases intentados ',
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


st.title("Identificación posición débil")

st.write("Puedes usar esta página para identificar que posición tienes más debil respecto al resto de equipos en las estadísticas que te interesn según el perfil de tu equipo.")

# Supongamos que 'df' es tu DataFrame
teams = df['team'].unique()
st.write("A continuación, seleccione el equipo que quiere analizar respecto al resto:")
selected_team = st.selectbox('Selecciona un equipo:', teams)
st.session_state.selected_team = selected_team

st.write("Una vez seleccionado el equipo, seleccione las estadisticas a analizar y filtre según le interese para su análisis.")

# Inverso del diccionario de mapeo para buscar la clave original por el valor mapeado
estadisticas_inverso = {v: k for k, v in estadisticas.items()}
posiciones_inverso = {v: k for k, v in posiciones.items()}
ligas_inverso = {v: k for k, v in ligas.items()}


# Mapea los nombres de las columnas para mostrar en el selectbox
df_columns_mapeadas = [estadisticas.get(col, col) for col in df.select_dtypes(include=np.number).columns]
df_positions_mapeadas = [posiciones.get(pos, pos) for pos in df.pos.unique()]

# Selecciona la primera estadística
selected_stat1_mapeada = st.selectbox("Selecciona la primera estadística", df_columns_mapeadas)
selected_stat1 = estadisticas_inverso.get(selected_stat1_mapeada, selected_stat1_mapeada)

# Elimina la columna seleccionada de las opciones para el segundo selectbox
df_columns_mapeadas_sin_selected1 = [col for col in df_columns_mapeadas if col != selected_stat1_mapeada]

# Selecciona la segunda estadística
selected_stat2_mapeada = st.selectbox("Selecciona la segunda estadística", df_columns_mapeadas_sin_selected1)
selected_stat2 = estadisticas_inverso.get(selected_stat2_mapeada, selected_stat2_mapeada)

filtered_df = pd.DataFrame()
selected_positions_mapeadas = st.multiselect("Selecciona las posiciones", df_positions_mapeadas)
selected_positions = [posiciones_inverso.get(pos, pos) for pos in selected_positions_mapeadas]

if selected_positions:    
    filtered_df = df[df['pos'].isin(selected_positions)]

    seasons = filtered_df['season'].unique()
    selected_season = st.selectbox("Selecciona una temporada", seasons)
    if selected_season:
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
        print(filtered_df.league.unique())
        df_ligas_mapeadas = [ligas.get(liga, liga) for liga in filtered_df.league.unique()]
        
        selected_leagues_mapeadas = st.multiselect("Selecciona las ligas", df_ligas_mapeadas)
        selected_leagues = [ligas_inverso.get(liga, liga) for liga in selected_leagues_mapeadas]

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