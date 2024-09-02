import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Example data (can be replaced with real data)
# Load data

# Page configuration
st.set_page_config(
    page_title="Welcome",
    page_icon="⚽",
    layout="centered"
)
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit_app/data.csv')
    df = df[df.season != '2023-2024']
    return df
df = load_data()
# Welcome page
st.title("Welcome to the Player Performance Analysis Tool")
st.image("streamlit_app/imagen.png")

st.markdown("""
### What can you do with this tool?
This application allows you to perform detailed analyses of player performance across seasons. You will be able to answer questions such as:
- In which positions does your team have less potential compared to its rivals?
- What are the weaknesses of your rivals?
- What has been the detailed performance of a specific player?
- Which players are most similar to a specific player?
- What is the performance prediction for a player for the next season and why?

### Adjustable Parameters
You will be able to adjust various parameters to customize your analyses, including:
- Selection of specific statistics
- Filters by team, league, season, player age, and more
- Configuration of prediction models

### Sample Data
Below, you can see a sample of the data you will be working with:
""")

# Display a sample of the data
st.write(df[['player', 'season', 'team', 'pos'] + [col for col in df.columns if col not in ['player', 'season', 'team', 'pos']]].sample(10))

# Comment to add images or external resources
# You can add a welcome image using st.image('image_path.jpg')
# Brief explanation of the tool's functionalities
st.markdown("""
### Application Tabs

#### Identification of Weak Points
In this section, you can compare your team with others to identify areas for improvement.

#### Player Details
Analyze in depth the performance of a specific player throughout the last season.

#### Player Comparison
Find players similar to the one you are analyzing and compare their statistics to make informed decisions.

#### Performance Prediction
Use prediction models to estimate future player performance and make strategic decisions.
""")

st.markdown("""
### Ready to get started?
Navigate through the pages on the left sidebar to begin your analysis.
""")

# Comment for the developer
# You can further customize the interface according to the specific needs of the user and the available data.
