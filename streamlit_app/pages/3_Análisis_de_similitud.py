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
    page_title="Similarity Analysis",
    page_icon="📝",
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
    'dispossessed': 'Ball Losses',
    'accurateFinalThirdPasses': 'Accurate Passes in the Final Third',
    'bigChancesCreated': 'Big Chances Created',
    'keyPasses': 'Key Passes (SofaScore)',
    'accurateCrosses': 'Accurate Crosses',
    'accurateCrossesPercentage': 'Accurate Crosses Percentage',
    'accurateLongBalls': 'Accurate Long Balls',
    'accurateLongBallsPercentage': 'Accurate Long Balls Percentage',
    'dribbledPast': 'Dribbled Past',
    'bigChancesMissed': 'Big Chances Missed',
    'hitWoodwork': 'Shots Hit the Woodwork',
    'errorLeadToGoal': 'Errors Leading to Goals',
    'errorLeadToShot': 'Errors Leading to Shots',
    'passToAssist': 'Passes to Assist',
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
    '90s': 'Complete Matches (90 Minutes)',
    'Gls': 'Goals',
    'Ast': 'Assists',
    'G+A': 'Goals + Assists',
    'G-PK': 'Goals (excluding Penalties)',
    'PK': 'Penalties Scored',
    'PKatt': 'Penalties Attempted',
    'CrdY': 'Yellow Cards',
    'CrdR': 'Red Cards',
    'xG': 'Expected Goals (xG)',
    'npxG': 'Non-Penalty Expected Goals (npxG)',
    'xAG': 'Expected Assists (xAG)',
    'npxG+xAG': 'Non-Penalty Expected Goals + Expected Assists (npxG+xAG)',
    'PrgC': 'Completed Progressive Passes',
    'PrgP': 'Progressive Passes',
    'PrgR': 'Progressive Runs',
    'Gls_90': 'Goals per 90 Minutes',
    'Ast_90': 'Assists per 90 Minutes',
    'G+A_90': 'Goals + Assists per 90 Minutes',
    'G-PK_90': 'Goals (excluding Penalties) per 90 Minutes',
    'G+A-PK_90': 'Goals + Assists (excluding Penalties) per 90 Minutes',
    'xG_90': 'Expected Goals per 90 Minutes (xG)',
    'xAG_90': 'Expected Assists per 90 Minutes (xAG)',
    'xG+xAG_90': 'Expected Goals + Expected Assists per 90 Minutes (xG+xAG)',
    'npxG_90': 'Non-Penalty Expected Goals per 90 Minutes (npxG)',
    'npxG+xAG_90': 'Non-Penalty Expected Goals + Expected Assists per 90 Minutes (npxG+xAG)',
    'Sh': 'Shots',
    'SoT': 'Shots on Target',
    'SoT%': 'Shots on Target Percentage',
    'Sh/90': 'Shots per 90 Minutes',
    'SoT/90': 'Shots on Target per 90 Minutes',
    'G/Sh': 'Goals per Shot',
    'G/SoT': 'Goals per Shot on Target',
    'Dist': 'Average Shot Distance',
    'FK': 'Free Kicks',
    'npxG/Sh': 'Non-Penalty Expected Goals per Shot (npxG/Sh)',
    'G-xG': 'Goals - Expected Goals (G-xG)',
    'np:G-xG': 'Non-Penalty Goals - Non-Penalty Expected Goals (np:G-xG)',
    'Total_Cmp': 'Completed Passes',
    'Total_Att': 'Total Passes Attempted',
    'Total_Cmp%': 'Pass Completion Percentage',
    'Total_TotDist': 'Total Pass Distance',
    'Total_PrgDist': 'Progressive Pass Distance',
    'Short_Cmp': 'Completed Short Passes',
    'Short_Att': 'Short Passes Attempted',
    'Short_Cmp%': 'Short Pass Completion Percentage',
    'Medium_Cmp': 'Completed Medium Passes',
    'Medium_Att': 'Medium Passes Attempted',
    'Medium_Cmp%': 'Medium Pass Completion Percentage',
    'Long_Cmp': 'Completed Long Passes',
    'Long_Att': 'Long Passes Attempted',
    'Long_Cmp%': 'Long Pass Completion Percentage',
    'xA': 'Expected Assists (xA)',
    'A-xAG': 'Assists - Expected Assists (A-xAG)',
    'KP': 'Key Passes (FBref)',
    '1/3': 'Passes to Final Third',
    'PPA': 'Passes to Penalty Area',
    'CrsPA': 'Crosses to Penalty Area',
    'Att': 'Passes Attempted',
    'Pass Types_Live': 'Passes in Play',
    'Pass Types_Dead': 'Set-Piece Passes',
    'Pass Types_FK': 'Free Kick Passes',
    'Pass Types_TB': 'Through Balls',
    'Pass Types_Sw': 'Switches of Play',
    'Pass Types_Crs': 'Crosses',
    'Pass Types_TI': 'Throw-Ins',
    'Pass Types_CK': 'Corners',
    'Corner Kicks_In': 'Corners to Near Post',
    'Corner Kicks_Out': 'Corners to Far Post',
    'Corner Kicks_Str': 'Corners to Center',
    'Outcomes_Cmp': 'Completed Actions',
    'Outcomes_Off': 'Offensive Actions',
    'Outcomes_Blocks': 'Blocked Actions',
    'SCA': 'Actions Leading to Shots (SCA)',
    'SCA90': 'Actions Leading to Shots per 90 Minutes (SCA90)',
    'SCA_PassLive': 'Actions Leading to Shots in Play',
    'SCA_PassDead': 'Actions Leading to Shots from Set-Pieces',
    'SCA_TO': 'Actions Leading to Shots from Turnovers',
    'SCA_Sh': 'Actions Leading to Shots from Shots',
    'SCA_Fld': 'Actions Leading to Shots from Fouls',
    'SCA_Def': 'Actions Leading to Shots from Defensive Actions',
    'GCA': 'Actions Leading to Goals (GCA)',
    'GCA90': 'Actions Leading to Goals per 90 Minutes (GCA90)',
    'GCA_PassLive': 'Actions Leading to Goals in Play',
    'GCA_PassDead': 'Actions Leading to Goals from Set-Pieces',
    'GCA_TO': 'Actions Leading to Goals from Turnovers',
    'GCA_Sh': 'Actions Leading to Goals from Shots',
    'GCA_Fld': 'Actions Leading to Goals from Fouls',
    'GCA_Def': 'Actions Leading to Goals from Defensive Actions',
    'Tkl': 'Tackles',
    'TklW': 'Tackles Won',
    'Tackles_Def 3rd': 'Defensive Third Tackles',
    'Tackles_Mid 3rd': 'Midfield Tackles',
    'Tackles_Att 3rd': 'Attacking Third Tackles',
    'Chall_Tkl': 'Challenges Won',
    'Chall_Att': 'Challenges Attempted',
    'Chall_Tkl%': 'Challenges Won Percentage',
    'Chall_Lost': 'Challenges Lost',
    'Blocks': 'Blocks',
    'Blocks_Sh': 'Shot Blocks',
    'Blocks_Pass': 'Pass Blocks',
    'Int': 'Interceptions',
    'Tkl+Int': 'Tackles + Interceptions',
    'Clr': 'Clearances',
    'Err': 'Errors',
    'Touches': 'Touches',
    'Touches_Def Pen': 'Defensive Area Touches',
    'Touches_Def 3rd': 'Defensive Third Touches',
    'Touches_Mid 3rd': 'Midfield Touches',
    'Touches_Att 3rd': 'Attacking Third Touches',
    'Touches_Att Pen': 'Attacking Area Touches',
    'Touches_Live': 'Touches in Play',
    'Take-Ons_Att': 'Take-Ons Attempted',
    'Take-Ons_Succ': 'Successful Take-Ons',
    'Take-Ons_Succ%': 'Successful Take-Ons Percentage',
    'Take-Ons_Tkld': 'Failed Take-Ons',
    'Take-Ons_Tkld%': 'Failed Take-Ons Percentage',
    'Carries': 'Carries',
    'Carries_TotDist': 'Total Carry Distance',
    'Carries_PrgDist': 'Progressive Carry Distance',
    'Carries_PrgC': 'Progressive Carries',
    'Carries_1/3': 'Carries to Final Third',
    'Carries_CPA': 'Carries to Penalty Area',
    'Carries_Mis': 'Failed Carries',
    'Carries_Dis': 'Lost Carries',
    'Receiving_Rec': 'Receipts',
    'Receiving_PrgR': 'Progressive Receipts',
    'Mn/MP': 'Minutes per Match',
    'Min%': 'Minutes Played Percentage',
    'Compl': 'Complete Matches',
    'Subs': 'Substitutions',
    'unSub': 'Not Substituted',
    'Team_Succ_PPM': 'Team Points per Match',
    'Team_Succ_onG': 'Team Goals Scored',
    'Team_Succ_onGA': 'Team Goals Against',
    'Team_Succ_+/-': 'Team Goal Difference',
    'Team_Succ_+/-90': 'Team Goal Difference per 90 Minutes',
    'Team_Succ_On-Off': 'Team Effectiveness with/without Player',
    'Team_Succ_onxG': 'Team Expected Goals For',
    'Team_Succ_onxGA': 'Team Expected Goals Against',
    'Team_Succ_xG+/-': 'Team Expected Goals Difference',
    'Team_Succ_xG+/-90': 'Team Expected Goals Difference per 90 Minutes',
    'Team_Succ_On-Off.1': 'Team Effectiveness with/without Player (Variant)',
    '2CrdY': 'Double Yellow Card',
    'Fls': 'Fouls Committed',
    'Fld': 'Fouls Received (FBref)',
    'Off': 'Offside',
    'Crs': 'Crosses',
    'PKwon': 'Penalties Won',
    'PKcon': 'Penalties Conceded',
    'OG': 'Own Goals',
    'Recov': 'Recoveries',
    'Aerial_Won': 'Aerial Duels Won',
    'Aerial_Lost': 'Aerial Duels Lost',
    'Aerial_Won%': 'Aerial Duels Won Percentage (FBref)',
    'Gen. Role': 'General Role',
    'Role': 'Specific Role',
    'xGoalsAdded': 'Expected Goals Added',
    'xGoalsAdded_p90': 'Expected Goals Added per 90 Minutes',
    'DAVIES': 'DAVIES',
    'DAVIES_Box Activity': 'Box Activity (DAVIES)',
    'DAVIES_Shooting': 'Shooting (DAVIES)',
    'DAVIES_Final Ball': 'Final Ball (DAVIES)',
    'DAVIES_Dribbles and Carries': 'Dribbles and Carries (DAVIES)',
    'DAVIES_Buildup Passing': 'Buildup Passing (DAVIES)',
    'DAVIES_Defense': 'Defense (DAVIES)',
    'DAVIES_p90': 'DAVIES per 90 Minutes',
    'DAVIES_Box Activity_p90': 'Box Activity per 90 Minutes (DAVIES)',
    'DAVIES_Shooting_p90': 'Shooting per 90 Minutes (DAVIES)',
    'DAVIES_Final Ball_p90': 'Final Ball per 90 Minutes (DAVIES)',
    'DAVIES_Dribbles and Carries_p90': 'Dribbles and Carries per 90 Minutes (DAVIES)',
    'DAVIES_Buildup Passing_p90': 'Buildup Passing per 90 Minutes (DAVIES)',
    'DAVIES_Defense_p90': 'Defense per 90 Minutes (DAVIES)',
    'team_elo': 'Team ELO',
    'team_rank': 'Team Ranking'
}

# Function to load the data
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season != '2023-2024']
    df_no_dav = pd.read_csv('streamlit_app/model_output_no_dav.csv')
    df_dav = pd.read_csv('streamlit_app/model_output_dav.csv')
    return df, df_no_dav, df_dav

# Function to normalize selected features
def normalize_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def calculate_percentile(df, features):
    percentiles_df = df[features].apply(lambda x: x if x.dtype == 'object' else x.rank(pct=True))
    return percentiles_df

# Function to find similar players
def find_similar_players(df, target_player, target_team, features, top_n=10, leagues=None, teams=None, age_range=None):
    # Filter only numerical features
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    df_normalized = normalize_features(df.copy(), numeric_features)
    target_position = df[(df['player'] == target_player.player) & (df['team'] == target_team) & (df['season'] == '2022-2023')]['pos'].values[0].split(",")
    target_position = [pos.strip() for pos in target_position]  # Remove extra spaces

    df = df[df['pos'].apply(lambda x: any(pos in x.split(",") for pos in target_position))]
    target_vector = df_normalized[(df_normalized['player'] == target_player.player) & (df_normalized['team'] == target_team) & (df['season'] == '2022-2023')][numeric_features].values
    if leagues:
        df = df[df['league'].isin(leagues)]
    if teams:
        df = df[df['team'].isin(teams)]
    if age_range:
        df = df[df['age'] <= age_range]

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(df_normalized[numeric_features], target_vector)
    # Add similarity to dataframe and convert to percentage
    df_normalized['similarity'] = similarity_matrix[:, 0] * 100
    df['similarity'] = df_normalized.loc[df.index].similarity

    # Filter players according to optional filters
    similar_players = df[(df['season'] == '2022-2023') & (df['player'] != target_player.player)]
    
    similar_players = similar_players.sort_values(by='similarity', ascending=False).head(top_n)
    
    return similar_players

# Function to create a radar plot
def create_radar_plot(selected_player, player_data, features):
    # Create Radar object
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
    fig, ax = radar.plot_radar(ranges=[(0,1) for feature in features], 
                            params=[statistics.get(col, col) for col in features], 
                            values=[selected_player[features].values.tolist()[0], player_data[features].values.tolist()[0]], 
                            radar_color=['#B6282F', '#344D94'], 
                            title=title, alphas=[0.55, 0.5],
                            compare=True)
    st.pyplot(fig)
    
# Function to create player history
def create_player_history(df, selected_player_name, player_name, selected_features):
    # Filter DataFrame to get the history of the specified player
    player_history = df[df['player'] == player_name]
    selected_player_history = df[df['player'] == selected_player_name]
    relevant_seasons = player_history['season'].unique()
    selected_player_history = selected_player_history[selected_player_history.season.isin(relevant_seasons)]
    player_history = player_history[player_history.season.isin(relevant_seasons)]
    df = df[df.season.isin(relevant_seasons)]

    if player_history.empty:
        st.write(f"No history found for player {player_name}.")
        return

    # Get the player's position
    player_position = player_history['pos'].iloc[0]

    for i, feature in enumerate(selected_features):
        if feature in df.columns:
            feature_mapped = statistics.get(feature, feature)
            fig = go.Figure()
            # Filter rows of the player that have a value for the current statistic
            player_feature_history = player_history
            
            if not player_feature_history.empty:
                # Add player's points
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

                # Calculate and add medians by season and position
                medians = df[(df['pos'] == player_position)].groupby('season')[feature].median().reset_index()

                # Add the median line by position
                fig.add_trace(go.Scatter(
                    x=medians['season'],
                    y=medians[feature],
                    mode='lines+markers+text',
                    text=[f'{val:.2f}' for val in medians[feature]],
                    textposition='top center',
                    name=f'Median for position - {feature_mapped}',
                    legendgroup=f'Median for position - {feature_mapped}',
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
                    text=f"No history found for player {player_name} in the feature '{feature_mapped}'.",
                    showlegend=False
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='text',
                text=f"The feature '{feature_mapped}' is not in the DataFrame.",
                showlegend=False
            ))

        # Set the labels and title of the chart
        fig.update_layout(
            title=f'Historical Comparison of {feature_mapped}',
            xaxis_title='Season',
            yaxis_title='Value',
            legend_title='Legend',
            legend=dict(
                x=1,  # Horizontal position (0: left, 1: right)
                y=1,  # Vertical position (0: bottom, 1: top)
                orientation='v'  # Legend orientation (options: 'v', 'h')
            )
        )

        st.plotly_chart(fig)


def stateful_button(*args, key=None, sidebar=False, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False
    if sidebar == False:
        if st.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]
    else:
        if st.sidebar.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]
    return st.session_state[key]

def select_features(df_columns_mapped):
    if 'selected_features' not in st.session_state:
        # If no features are selected in the session, set them to an empty list
        st.session_state['selected_features'] = []


    # Selection of numerical features
    if stateful_button('Select all features.', sidebar=True, key="select_features"):
        st.session_state.selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    default_cols = [statistics.get(feature, feature) for feature in st.session_state.selected_features]
    st.session_state.selected_features = st.sidebar.multiselect('Select features', df_columns_mapped, default=default_cols)
    selected_features = st.session_state.selected_features

    return selected_features

statistics_inverse = {v: k for k, v in statistics.items()}
positions_inverse = {v: k for k, v in positions.items()}
leagues_inverse = {v: k for k, v in leagues.items()}

# Load data
df, df_no_dav, df_dav = load_data()
target_player = st.session_state.player
# Application title
st.title('Search for Similar Players')
st.write(f"On this page, you can view players with a profile similar to {target_player.player} to find the ideal replacement.")
# Columns for main filters
selected_player = st.session_state.player
selected_team = st.session_state.player.team

# Feature selection and additional filters
st.header('Additional Filters')
col3, col4 = st.columns(2)
with col3:
    filter_by_league = st.checkbox('Filter by leagues')
    if filter_by_league:
        df_leagues_mapped = [leagues.get(league, league) for league in df.league.unique()]
        selected_leagues_mapped = st.multiselect('Select leagues', df_leagues_mapped)
        selected_leagues = [leagues_inverse.get(league, league) for league in selected_leagues_mapped]
        filtered_df = df[df['league'].isin(selected_leagues)]
    else:
        selected_leagues = None
    
    filter_by_team = st.checkbox('Filter by teams')
    if filter_by_team:
        selected_teams = st.multiselect('Select teams', filtered_df['team'].unique())
        filtered_df = df[df['team'].isin(selected_teams)]
    else:
        selected_teams = None
with col4:
    filter_by_age = st.checkbox('Filter by age range')
    if filter_by_age:
        age_range = st.slider('Select age range', min_value=int(filtered_df['age'].min()), max_value=int(filtered_df['age'].max()))
        filtered_df = df[df['age'] <= age_range]
    else:
        age_range = None
df_columns_mapped = [statistics.get(col, col) for col in df.select_dtypes(include=[np.number]).columns.tolist()]
selected_features = select_features(df_columns_mapped)

# Display only a few selected features and a counter
MAX_DISPLAYED_FEATURES = 5
if len(selected_features) <= MAX_DISPLAYED_FEATURES:
    selected_features_display = ", ".join(selected_features)
else:
    selected_features_display = ", ".join(selected_features[:MAX_DISPLAYED_FEATURES]) + f", and {len(selected_features) - MAX_DISPLAYED_FEATURES} more"

st.write(f"Selected features: {selected_features_display}")
selected_features = [statistics_inverse.get(feature, feature) for feature in st.session_state.selected_features]

# Slider to select the number of similar players to display
top_n = st.slider('Number of similar players to display', 1, 20, 10)

# Button to run the analysis
if selected_team and len(selected_features) > 0:
    similar_players = find_similar_players(df, selected_player, selected_team, selected_features, top_n, selected_leagues, selected_teams, age_range)
    st.header(f'Similar Players to {selected_player.player} in {selected_team}')
    selected_player_data = df[(df['player'] == selected_player.player) & (df['season'] == '2022-2023') & (df.team == selected_team)]
    for i, row in similar_players.iterrows():
        st.write(f"Player: {row['player']} | Team: {row['team']} | Similarity: {row['similarity']:.2f}%")

    # Show visual list of similar players with similarity percentage
    similar_players_list = []
    for i, row in similar_players.iterrows():
        similar_players_list.append(f"{row['player']} | {row['team']} | Age: {row['age']} | Similarity: {row['similarity']:.2f}%")

    selected_similar_player = st.selectbox('Select a player for a more detailed comparison.', similar_players_list)
    if selected_similar_player is not None:
        # Button to show the full table
        similar_players = pd.concat([selected_player_data, similar_players])
        if stateful_button('Show table with statistics.', key="similar_players"):
        #if st.button('Show full details'):
            st.write(similar_players[['player', 'team', 'pos'] + ['similarity'] + selected_features ])
        
        # Create radar plot for the selected player
        similar_player_name = selected_similar_player.split('|')[0].strip()
        df_normalized = calculate_percentile(df.copy(), ['player', 'team', 'season'] + selected_features)
        if len(selected_features) < 15 and len(selected_features) >= 3:
            similar_player_data_norm = df_normalized[(df_normalized['player'] == similar_player_name) & (df_normalized['season'] == '2022-2023')]
            selected_player_data_norm = df_normalized[(df_normalized['player'] == selected_player.player) & (df_normalized['season'] == '2022-2023') & (df_normalized.team == selected_team)]
            create_radar_plot(selected_player_data_norm, similar_player_data_norm, selected_features)
            # Create history for the selected player
            st.header("Historical data for the selected player.")
            create_player_history(df, selected_player.player, similar_player_name, selected_features)
        else:
            st.write("Please select fewer than 15 or more than 3 features to view a radar plot and the historical data for the chosen player across different features.")
        st.session_state.similar_player = similar_player_name
        st.session_state.similar_team = selected_similar_player.split('|')[1].strip()
        st.write(f"By pressing the button below, you will be redirected to a page where you can get a prediction of {similar_player_name}'s performance in the next season.")
        st.page_link("pages/4_Performance_Prediction.py", label="Button")
    else:
        st.write(f"No sufficiently similar player found to {target_player.player}. Select a different combination of features or be less strict with the filters.")
else:
    st.write(f"Please select at least 3 features to find the player that most closely resembles {target_player.player} based on the selected statistics.")
    st.write("You can also click the button in the sidebar to select all features.")

