import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(
page_title="Player Details",
page_icon="🔎",
layout="wide",
initial_sidebar_state="expanded")


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

# Statistics mapping dictionary
statistics = {
    'groundDuelsWon': 'Ground Duels Won',
    'groundDuelsWonPercentage': 'Ground Duels Won Percentage',
    'aerialDuelsWonPercentage': 'Aerial Duels Won Percentage (SofaScore)',
    'wasFouled': 'Fouls Received (SofaScore)',
    'dispossessed': 'Dispossessed',
    'accurateFinalThirdPasses': 'Accurate Final Third Passes',
    'bigChancesCreated': 'Big Chances Created',
    'keyPasses': 'Key Passes (SofaScore)',
    'accurateCrosses': 'Accurate Crosses',
    'accurateCrossesPercentage': 'Accurate Crosses Percentage',
    'accurateLongBalls': 'Accurate Long Balls',
    'accurateLongBallsPercentage': 'Accurate Long Balls Percentage',
    'dribbledPast': 'Dribbled Past',
    'bigChancesMissed': 'Big Chances Missed',
    'hitWoodwork': 'Hit Woodwork',
    'errorLeadToGoal': 'Errors Leading to Goal',
    'errorLeadToShot': 'Errors Leading to Shot',
    'passToAssist': 'Pass to Assist',
    'player': 'Player',
    'team': 'Team',
    'season': 'Season',
    'league': 'League',
    'nation': 'Nation',
    'pos': 'Position',
    'age': 'Age',
    'born': 'Birth',
    'MP': 'Matches Played',
    'Starts': 'Starts',
    'Min': 'Minutes Played',
    '90s': 'Full Matches (90 minutes)',
    'Gls': 'Goals',
    'Ast': 'Assists',
    'G+A': 'Goals + Assists',
    'G-PK': 'Goals (excluding penalties)',
    'PK': 'Penalties Scored',
    'PKatt': 'Penalties Attempted',
    'CrdY': 'Yellow Cards',
    'CrdR': 'Red Cards',
    'xG': 'Expected Goals (xG)',
    'npxG': 'Non-Penalty Expected Goals (npxG)',
    'xAG': 'Expected Assists (xAG)',
    'npxG+xAG': 'Non-Penalty Expected Goals + Expected Assists (npxG+xAG)',
    'PrgC': 'Progressive Passes Completed',
    'PrgP': 'Progressive Passes',
    'PrgR': 'Progressive Runs',
    'Gls_90': 'Goals per 90 minutes',
    'Ast_90': 'Assists per 90 minutes',
    'G+A_90': 'Goals + Assists per 90 minutes',
    'G-PK_90': 'Goals (excluding penalties) per 90 minutes',
    'G+A-PK_90': 'Goals + Assists (excluding penalties) per 90 minutes',
    'xG_90': 'Expected Goals per 90 minutes (xG)',
    'xAG_90': 'Expected Assists per 90 minutes (xAG)',
    'xG+xAG_90': 'Expected Goals + Expected Assists per 90 minutes (xG+xAG)',
    'npxG_90': 'Non-Penalty Expected Goals per 90 minutes (npxG)',
    'npxG+xAG_90': 'Non-Penalty Expected Goals + Expected Assists per 90 minutes (npxG+xAG)',
    'Sh': 'Shots',
    'SoT': 'Shots on Target',
    'SoT%': 'Shots on Target Percentage',
    'Sh/90': 'Shots per 90 minutes',
    'SoT/90': 'Shots on Target per 90 minutes',
    'G/Sh': 'Goals per Shot',
    'G/SoT': 'Goals per Shot on Target',
    'Dist': 'Average Shot Distance',
    'FK': 'Free Kicks',
    'npxG/Sh': 'Non-Penalty Expected Goals per Shot (npxG/Sh)',
    'G-xG': 'Goals - Expected Goals (G-xG)',
    'np:G-xG': 'Non-Penalty Goals - Non-Penalty Expected Goals (np:G-xG)',
    'Total_Cmp': 'Completed Passes',
    'Total_Att': 'Total Passes Attempted',
    'Total_Cmp%': 'Completed Passes Percentage',
    'Total_TotDist': 'Total Pass Distance',
    'Total_PrgDist': 'Progressive Pass Distance',
    'Short_Cmp': 'Completed Short Passes',
    'Short_Att': 'Attempted Short Passes',
    'Short_Cmp%': 'Completed Short Passes Percentage',
    'Medium_Cmp': 'Completed Medium Passes',
    'Medium_Att': 'Attempted Medium Passes',
    'Medium_Cmp%': 'Completed Medium Passes Percentage',
    'Long_Cmp': 'Completed Long Passes',
    'Long_Att': 'Attempted Long Passes',
    'Long_Cmp%': 'Completed Long Passes Percentage',
    'xA': 'Expected Assists (xA)',
    'A-xAG': 'Assists - Expected Assists (A-xAG)',
    'KP': 'Key Passes (FBref)',
    '1/3': 'Passes to Final Third',
    'PPA': 'Passes to Penalty Area',
    'CrsPA': 'Crosses to Penalty Area',
    'Att': 'Passes Attempted',
    'Pass Types_Live': 'Passes in Play',
    'Pass Types_Dead': 'Set Piece Passes',
    'Pass Types_FK': 'Free Kick Passes',
    'Pass Types_TB': 'Through Balls',
    'Pass Types_Sw': 'Switches of Play',
    'Pass Types_Crs': 'Crosses',
    'Pass Types_TI': 'Throw-Ins',
    'Pass Types_CK': 'Corners',
    'Corner Kicks_In': 'Corners Near Post',
    'Corner Kicks_Out': 'Corners Far Post',
    'Corner Kicks_Str': 'Corners Center',
    'Outcomes_Cmp': 'Completed Actions',
    'Outcomes_Off': 'Offensive Actions',
    'Outcomes_Blocks': 'Blocked Actions',
    'SCA': 'Shot-Creating Actions (SCA)',
    'SCA90': 'Shot-Creating Actions per 90 minutes (SCA90)',
    'SCA_PassLive': 'Shot-Creating Actions in Play',
    'SCA_PassDead': 'Shot-Creating Actions from Set Pieces',
    'SCA_TO': 'Shot-Creating Actions from Turnovers',
    'SCA_Sh': 'Shot-Creating Actions from Shots',
    'SCA_Fld': 'Shot-Creating Actions from Fouls',
    'SCA_Def': 'Shot-Creating Actions from Defensive Actions',
    'GCA': 'Goal-Creating Actions (GCA)',
    'GCA90': 'Goal-Creating Actions per 90 minutes (GCA90)',
    'GCA_PassLive': 'Goal-Creating Actions in Play',
    'GCA_PassDead': 'Goal-Creating Actions from Set Pieces',
    'GCA_TO': 'Goal-Creating Actions from Turnovers',
    'GCA_Sh': 'Goal-Creating Actions from Shots',
    'GCA_Fld': 'Goal-Creating Actions from Fouls',
    'GCA_Def': 'Goal-Creating Actions from Defensive Actions',
    'Tkl': 'Tackles',
    'TklW': 'Tackles Won',
    'Tackles_Def 3rd': 'Tackles in Defensive Third',
    'Tackles_Mid 3rd': 'Tackles in Midfield Third',
    'Tackles_Att 3rd': 'Tackles in Attacking Third',
    'Chall_Tkl': 'Challenges Won',
    'Chall_Att': 'Challenges Attempted',
    'Chall_Tkl%': 'Challenge Win Percentage',
    'Chall_Lost': 'Challenges Lost',
    'Blocks': 'Blocks',
    'Blocks_Sh': 'Shot Blocks',
    'Blocks_Pass': 'Pass Blocks',
    'Clear': 'Clearances',
    'Inter': 'Interceptions',
    'AerialsWon': 'Aerial Duels Won',
    'AerialsLost': 'Aerial Duels Lost',
    'AerialsWon%': 'Aerial Duels Won Percentage',
    'CleanS': 'Clean Sheets',
    'Saves': 'Saves',
    'Saves%': 'Save Percentage',
    'Errors': 'Errors Leading to Goals',
    'ErrorsShot': 'Errors Leading to Shots',
    'ErrorsDef': 'Defensive Errors',
    'DuelWon': 'Duels Won',
    'DuelLost': 'Duels Lost',
    'DuelWon%': 'Duels Won Percentage',
    'Pressure': 'Pressures',
    'PressureWon': 'Pressures Won',
    'PressureLost': 'Pressures Lost',
    'PressureWon%': 'Pressures Won Percentage',
    'Pressures': 'Pressures',
    'Pressures_Won': 'Pressures Won',
    'Pressures_Lost': 'Pressures Lost',
    'Pressures_Won%': 'Pressures Won Percentage'
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
# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season != '2023-2024']
    return df

def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

df = load_data()

# Inverse of the mapping dictionary to look up the original key by the mapped value
statistics_inverse = {v: k for k, v in statistics.items()}
positions_inverse = {v: k for k, v in positions.items()}
leagues_inverse = {v: k for k, v in leagues.items()}

team_of_player = st.session_state.selected_team
st.title(f"Player Details")
st.write(f"Choose a player from your team ({team_of_player}) to analyze and compare with other players later.")
last_season_df = df[(df.season == '2022-2023') & (df.team == team_of_player)].drop('season', axis=1)
selected_player = st.selectbox("Choose a player", last_season_df['player'].unique())
player = last_season_df[(last_season_df['player'] == selected_player) & (last_season_df['team'] == team_of_player)]
st.header("Player Bio")
# Player bio information
player_name = player['player'].values[0]
player_position = player['pos'].values[0]
player_position = positions.get(player_position, player_position)

player_age = player['age'].values[0]
player_nation = player['nation'].values[0]
player_starts = player['Starts'].values[0]

# HTML to display player data
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
        <div class="bio-stat-heading">Nationality</div>
        <div class="bio-stat-value" id="player-nation"><span class="white-text">{player_nation}</span></div>
    </div>
    <div class="bio-stat">
        <div class="bio-stat-heading">Matches Started</div>
        <div class="bio-stat-value" id="player-starts"><span class="white-text">{player_starts}</span></div>
    </div>
</div>
"""

# Display the player's bio table in Streamlit
st.markdown(html, unsafe_allow_html=True)
statistics_inverse = {v: k for k, v in statistics.items()}
player_position = positions_inverse.get(player_position, player_position)

df_columns_mapped = [statistics.get(col, col) for col in df.select_dtypes(include=np.number).columns]
selected_features_mapped = st.multiselect('Select Key Features:', df_columns_mapped)
selected_features = [statistics_inverse.get(feature, feature) for feature in selected_features_mapped]

if len(selected_features) > 0:
    position_data = df[df['pos'] == player_position][selected_features]
    player_data = df[df['player'] == selected_player]
    player_position = player_data['pos'].values[0]
    
    st.header("Key Statistics")
    st.write(f"The bar chart below has minimum and maximum values considering only players of {player_name}'s position.")

    for feature in selected_features:
        min_val = df[df['pos'] == player_position][feature].min()
        max_val = df[df['pos'] == player_position][feature].max()
        player_val = player_data[feature].values[0]
  
        # Calculate the player's value percentage relative to the maximum
        percentage = (player_val - min_val) / (max_val - min_val) * 100

        # Determine color based on percentage (from dark to light green)
        green_color = int(255 - (percentage * 1.27))
        red_color = int(percentage * 1.27)
        color = f"rgb({red_color}, {green_color}, 0)"
        
        feature = statistics.get(feature, feature)

        # Show progress bar with custom style and percentile
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="margin-bottom: 5px;">{feature}: {player_val} (Percentile: {percentage:.2f}%)</div>
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 10px;">{min_val:.2f}</div>
                <div style="position: relative; height: 30px; width: 100%; border-radius: 5px; background: #f0f0f0;">
                    <div style="width: {percentage}%; height: 100%; border-radius: 5px; background: {color};"></div>
                </div>
                <div style="margin-left: 10px;">{max_val:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.write("Click the button below if you want to search for similar players.")
numeric_features = df[df.columns].select_dtypes(include=[np.number]).columns.tolist()
df_normalized = normalize_features(df.copy(), numeric_features)
st.session_state.player = df_normalized.loc[player.index[0]]
st.page_link("pages/3_Similarity_Analysis.py", label="Button")
