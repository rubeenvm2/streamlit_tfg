import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
page_title="Weak Position Identification",
page_icon="📉",
layout="wide",
initial_sidebar_state="expanded")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season!='2023-2024']
    return df
df = load_data()

# Position mapping dictionary
positions = {
    'FW': 'Forward',
    'MF': 'Midfielder',
    'DF': 'Defender',
    'FW,MF': 'Forward-Midfielder',
    'MF,FW': 'Midfielder-Forward',
    'FW,DF': 'Forward-Defender',
    'DF,FW': 'Defender-Forward',
    'MF,DF': 'Midfielder-Defender',
    'DF,MF': 'Defender-Midfielder'
    # Add more positions as needed
}

leagues = {
    'espla liga': 'La Liga (ESP)',
    'gerbundesliga': 'Bundesliga (GER)',
    'engpremier league': 'Premier League (ENG)',
    'itaserie a': 'Serie A (ITA)',
    'fraligue ': 'Ligue 1 (FRA)'
    # Add more leagues as needed
}

# Diccionario de mapeo de estadísticas
statistics = {
    'groundDuelsWon': 'Ground duels won',
    'groundDuelsWonPercentage': 'Percentage of ground duels won',
    'aerialDuelsWonPercentage': 'Percentage of aerial duels won (SofaScore)',
    'wasFouled': 'Fouls received (SofaScore)',
    'dispossessed': 'Ball losses',
    'accurateFinalThirdPasses': 'Accurate passes in the final third',
    'bigChancesCreated': 'Big chances created',
    'keyPasses': 'Key passes (SofaScore)',
    'accurateCrosses': 'Accurate crosses',
    'accurateCrossesPercentage': 'Percentage of accurate crosses',
    'accurateLongBalls': 'Accurate long balls',
    'accurateLongBallsPercentage': 'Percentage of accurate long balls',
    'dribbledPast': 'Dribbled past',
    'bigChancesMissed': 'Big chances missed',
    'hitWoodwork': 'Shots hit the woodwork',
    'errorLeadToGoal': 'Errors leading to goals',
    'errorLeadToShot': 'Errors leading to shots',
    'passToAssist': 'Passes leading to assists',
    'player': 'Player',
    'team': 'Team',
    'season': 'Season',
    'league': 'League',
    'nation': 'Nation',
    'pos': 'Position',
    'age': 'Age',
    'born': 'Birth',
    'MP': 'Matches played',
    'Starts': 'Starts',
    'Min': 'Minutes played',
    '90s': 'Complete matches (90 minutes)',
    'Gls': 'Goals',
    'Ast': 'Assists',
    'G+A': 'Goals + Assists',
    'G-PK': 'Goals excluding penalties',
    'PK': 'Penalties scored',
    'PKatt': 'Penalties attempted',
    'CrdY': 'Yellow cards',
    'CrdR': 'Red cards',
    'xG': 'Expected goals (xG)',
    'npxG': 'Non-penalty expected goals (npxG)',
    'xAG': 'Expected assists (xAG)',
    'npxG+xAG': 'Non-penalty expected goals + Expected assists (npxG+xAG)',
    'PrgC': 'Completed progressive passes',
    'PrgP': 'Progressive passes',
    'PrgR': 'Progressive runs',
    'Gls_90': 'Goals per 90 minutes',
    'Ast_90': 'Assists per 90 minutes',
    'G+A_90': 'Goals + Assists per 90 minutes',
    'G-PK_90': 'Goals excluding penalties per 90 minutes',
    'G+A-PK_90': 'Goals + Assists excluding penalties per 90 minutes',
    'xG_90': 'Expected goals per 90 minutes (xG)',
    'xAG_90': 'Expected assists per 90 minutes (xAG)',
    'xG+xAG_90': 'Expected goals + Expected assists per 90 minutes (xG+xAG)',
    'npxG_90': 'Non-penalty expected goals per 90 minutes (npxG)',
    'npxG+xAG_90': 'Non-penalty expected goals + Expected assists per 90 minutes (npxG+xAG)',
    'Sh': 'Shots',
    'SoT': 'Shots on target',
    'SoT%': 'Percentage of shots on target',
    'Sh/90': 'Shots per 90 minutes',
    'SoT/90': 'Shots on target per 90 minutes',
    'G/Sh': 'Goals per shot',
    'G/SoT': 'Goals per shot on target',
    'Dist': 'Average shot distance',
    'FK': 'Free kicks',
    'npxG/Sh': 'Non-penalty expected goals per shot (npxG/Sh)',
    'G-xG': 'Goals minus expected goals (G-xG)',
    'np:G-xG': 'Goals excluding penalties minus non-penalty expected goals (np:G-xG)',
    'Total_Cmp': 'Completed passes',
    'Total_Att': 'Total passes attempted',
    'Total_Cmp%': 'Percentage of completed passes',
    'Total_TotDist': 'Total pass distance',
    'Total_PrgDist': 'Progressive pass distance',
    'Short_Cmp': 'Completed short passes',
    'Short_Att': 'Attempted short passes',
    'Short_Cmp%': 'Percentage of completed short passes',
    'Medium_Cmp': 'Completed medium passes',
    'Medium_Att': 'Attempted medium passes',
    'Medium_Cmp%': 'Percentage of completed medium passes',
    'Long_Cmp': 'Completed long passes',
    'Long_Att': 'Attempted long passes',
    'Long_Cmp%': 'Percentage of completed long passes',
    'xA': 'Expected assists (xA)',
    'A-xAG': 'Assists minus expected assists (A-xAG)',
    'KP': 'Key passes (FBref)',
    '1/3': 'Passes to the final third',
    'PPA': 'Passes to the penalty area',
    'CrsPA': 'Crosses to the penalty area',
    'Att': 'Passes attempted',
    'Pass Types_Live': 'Passes in play',
    'Pass Types_Dead': 'Set piece passes',
    'Pass Types_FK': 'Free kick passes',
    'Pass Types_TB': 'Through balls',
    'Pass Types_Sw': 'Switches of play',
    'Pass Types_Crs': 'Crosses',
    'Pass Types_TI': 'Throw-ins',
    'Pass Types_CK': 'Corners',
    'Corner Kicks_In': 'Corners to the near post',
    'Corner Kicks_Out': 'Corners to the far post',
    'Corner Kicks_Str': 'Corners to the center',
    'Outcomes_Cmp': 'Completed actions',
    'Outcomes_Off': 'Offensive actions',
    'Outcomes_Blocks': 'Blocked actions',
    'SCA': 'Actions leading to a shot (SCA)',
    'SCA90': 'Actions leading to a shot per 90 minutes (SCA90)',
    'SCA_PassLive': 'Actions leading to a shot in play',
    'SCA_PassDead': 'Actions leading to a shot from set pieces',
    'SCA_TO': 'Actions leading to a shot from turnovers',
    'SCA_Sh': 'Actions leading to a shot from shots',
    'SCA_Fld': 'Actions leading to a shot from fouls',
    'SCA_Def': 'Actions leading to a shot from defense',
    'GCA': 'Actions leading to a goal (GCA)',
    'GCA90': 'Actions leading to a goal per 90 minutes (GCA90)',
    'GCA_PassLive': 'Actions leading to a goal in play',
    'GCA_PassDead': 'Actions leading to a goal from set pieces',
    'GCA_TO': 'Actions leading to a goal from turnovers',
    'GCA_Sh': 'Actions leading to a goal from shots',
    'GCA_Fld': 'Actions leading to a goal from fouls',
    'GCA_Def': 'Actions leading to a goal from defense',
    'Tkl': 'Tackles',
    'TklW': 'Tackles won',
    'Tackles_Def 3rd': 'Tackles in the defensive third',
    'Tackles_Mid 3rd': 'Tackles in the middle third',
    'Tackles_Att 3rd': 'Tackles in the attacking third',
    'Chall_Tkl': 'Challenges won',
    'Chall_Att': 'Challenges attempted',
    'Chall_Tkl%': 'Percentage of challenges won',
    'Chall_Lost': 'Challenges lost',
    'Blocks': 'Blocks',
    'Blocks_Sh': 'Shot blocks',
    'Blocks_Pass': 'Pass blocks',
    'Int': 'Interceptions',
    'Tkl+Int': 'Tackles + Interceptions',
    'Clr': 'Clearances',
    'Err': 'Errors',
    'Touches': 'Touches',
    'Touches_Def Pen': 'Touches in the defensive penalty area',
    'Touches_Def 3rd': 'Touches in the defensive third',
    'Touches_Mid 3rd': 'Touches in the middle third',
    'Touches_Att 3rd': 'Touches in the attacking third',
    'Touches_Att Pen': 'Touches in the attacking penalty area',
    'Touches_Live': 'Touches in play',
    'Take-Ons_Att': 'Take-ons attempted',
    'Take-Ons_Succ': 'Take-ons successful',
    'Take-Ons_Succ%': 'Percentage of successful take-ons',
    'Take-Ons_Tkld': 'Take-ons failed',
    'Take-Ons_Tkld%': 'Percentage of failed take-ons',
    'Carries': 'Carries',
    'Carries_TotDist': 'Total carry distance',
    'Carries_PrgDist': 'Progressive carry distance',
    'Carries_PrgC': 'Progressive carries',
    'Carries_1/3': 'Carries to the final third',
    'Carries_CPA': 'Carries to the penalty area',
    'Carries_Mis': 'Failed carries',
    'Carries_Dis': 'Lost carries',
    'Receiving_Rec': 'Receives',
    'Receiving_PrgR': 'Progressive receives',
    'Mn/MP': 'Minutes per match',
    'Min%': 'Percentage of minutes played',
    'Compl': 'Complete matches',
    'Subs': 'Substitutions',
    'unSub': 'Not substituted',
    'Team_Succ_PPM': 'Team points per match',
    'Team_Succ_onG': 'Team goals scored',
    'Team_Succ_onGA': 'Team goals conceded',
    'Team_Succ_+/-': 'Team goal difference',
    'Team_Succ_+/-90': 'Team goal difference per 90 minutes',
    'Team_Succ_On-Off': 'Team effectiveness with/without player',
    'Team_Succ_onxG': 'Team expected goals',
    'Team_Succ_onxGA': 'Team expected goals against',
    'Team_Succ_xG+/-': 'Team expected goals difference',
    'Team_Succ_xG+/-90': 'Team expected goals difference per 90 minutes',
    'Team_Succ_On-Off.1': 'Team effectiveness with/without player (variant)',
    '2CrdY': 'Double yellow cards',
    'Fls': 'Fouls committed',
    'Fld': 'Fouls received (FBref)',
    'Off': 'Offside',
    'Crs': 'Crosses',
    'PKwon': 'Penalties won',
    'PKcon': 'Penalties conceded',
    'OG': 'Own goals',
    'Recov': 'Recoveries',
    'Aerial_Won': 'Aerial duels won',
    'Aerial_Lost': 'Aerial duels lost',
    'Aerial_Won%': 'Percentage of aerial duels won (FBref)',
    'Gen. Role': 'General role',
    'Role': 'Specific role',
    'xGoalsAdded': 'Expected goals added',
    'xGoalsAdded_p90': 'Expected goals added per 90 minutes',
    'DAVIES': 'DAVIES',
    'DAVIES_Box Activity': 'Box activity (DAVIES)',
    'DAVIES_Shooting': 'Shooting (DAVIES)',
    'DAVIES_Final Ball': 'Final ball (DAVIES)',
    'DAVIES_Dribbles and Carries': 'Dribbles and carries (DAVIES)',
    'DAVIES_Buildup Passing': 'Build-up passing (DAVIES)',
    'DAVIES_Defense': 'Defense (DAVIES)',
    'DAVIES_p90': 'DAVIES per 90 minutes',
    'DAVIES_Box Activity_p90': 'Box activity per 90 minutes (DAVIES)',
    'DAVIES_Shooting_p90': 'Shooting per 90 minutes (DAVIES)',
    'DAVIES_Final Ball_p90': 'Final ball per 90 minutes (DAVIES)',
    'DAVIES_Dribbles and Carries_p90': 'Dribbles and carries per 90 minutes (DAVIES)',
    'DAVIES_Buildup Passing_p90': 'Build-up passing per 90 minutes (DAVIES)',
    'DAVIES_Defense_p90': 'Defense per 90 minutes (DAVIES)',
    'team_elo': 'Team ELO',
    'team_rank': 'Team ranking'
}


st.title("Identify Weak Position")

st.write("Use this page to identify which position is the weakest for your team compared to other teams, based on the statistics you are interested in.")

# Assuming 'df' is your DataFrame
teams = df['team'].unique()
st.write("Please select the team you want to analyze compared to the others:")
selected_team = st.selectbox('Select a team:', teams)
st.session_state.selected_team = selected_team

st.write("Once the team is selected, choose the statistics to analyze and filter as needed for your analysis.")

# Reverse mapping dictionaries to find the original key by the mapped value
stats_reverse = {v: k for k, v in estadisticas.items()}
positions_reverse = {v: k for k, v in posiciones.items()}
leagues_reverse = {v: k for k, v in ligas.items()}

# Map column names to show in the selectbox
df_columns_mapped = [estadisticas.get(col, col) for col in df.select_dtypes(include=np.number).columns]
df_positions_mapped = [posiciones.get(pos, pos) for pos in df.pos.unique()]

# Select the first statistic
selected_stat1_mapped = st.selectbox("Select the first statistic", df_columns_mapped)
selected_stat1 = stats_reverse.get(selected_stat1_mapped, selected_stat1_mapped)

# Remove the selected column from the options for the second selectbox
df_columns_mapped_without_selected1 = [col for col in df_columns_mapped if col != selected_stat1_mapped]

# Select the second statistic
selected_stat2_mapped = st.selectbox("Select the second statistic", df_columns_mapped_without_selected1)
selected_stat2 = stats_reverse.get(selected_stat2_mapped, selected_stat2_mapped)

filtered_df = pd.DataFrame()
selected_positions_mapped = st.multiselect("Select positions", df_positions_mapped)
selected_positions = [positions_reverse.get(pos, pos) for pos in selected_positions_mapped]

if selected_positions:    
    filtered_df = df[df['pos'].isin(selected_positions)]

    seasons = filtered_df['season'].unique()
    selected_season = st.selectbox("Select a season", seasons)
    if selected_season:
        filtered_df = filtered_df[filtered_df['season'] == selected_season]
        print(filtered_df.league.unique())
        df_leagues_mapped = [ligas.get(league, league) for league in filtered_df.league.unique()]
        
        selected_leagues_mapped = st.multiselect("Select leagues", df_leagues_mapped)
        selected_leagues = [leagues_reverse.get(league, league) for league in selected_leagues_mapped]

        if selected_leagues:
            filtered_df = filtered_df[filtered_df['league'].isin(selected_leagues)]
            
            teams = filtered_df[filtered_df['team'] != selected_team]['team'].unique()
            selected_teams = st.multiselect("Select teams", teams)
            
            if selected_teams:
                filtered_df = filtered_df[filtered_df['team'].isin(selected_teams)]

                min_age, max_age = st.slider("Select age range", int(filtered_df['age'].min()), int(filtered_df['age'].max()), (int(filtered_df['age'].min()), int(filtered_df['age'].max())))
                filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

                min_minutes, max_minutes = st.slider("Select minutes percentage range", int(filtered_df['Min%'].min()), int(filtered_df['Min%'].max()), (int(filtered_df['Min%'].min()), int(filtered_df['Min%'].max())))
                filtered_df = filtered_df[(filtered_df['Min%'] >= min_minutes) & (filtered_df['age'] <= max_minutes)]

if len(filtered_df) > 0:
    team_data = df[(df['team'] == selected_team) & (df.season == selected_season) & (df.pos.isin(selected_positions))]
    fig = go.Figure()
    
    # Scatter plot for the selected team data
    fig.add_trace(go.Scatter(
        x=team_data[selected_stat1],
        y=team_data[selected_stat2],
        mode='markers+text',
        text=team_data['player'],
        textposition='top center',
        marker=dict(color='orange', size=10),
        name=selected_team
    ))

    # Scatter plot for the filtered data
    fig.add_trace(go.Scatter(
        x=filtered_df[selected_stat1],
        y=filtered_df[selected_stat2],
        mode='markers',
        marker=dict(color='white', opacity=0.6),
        name='Other teams'
    ))

    # Configure labels and chart title
    fig.update_layout(
        title=f'{estadisticas.get(selected_stat1, selected_stat1)} vs {estadisticas.get(selected_stat2, selected_stat2)}',
        xaxis_title=estadisticas.get(selected_stat1, selected_stat1),
        yaxis_title=estadisticas.get(selected_stat2, selected_stat2),
        legend_title='Legend'
    )

    st.plotly_chart(fig)
