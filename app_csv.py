##### Streamlit Python Application - CSV #####

# used libraries 
import os
import json
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List
import hashlib
# solr library
import unidecode

# bing library for automation image
from bing_image_urls import bing_image_urls

# streamlit libraries
import streamlit as st 
from streamlit_searchbox import st_searchbox

# cosine similarity libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# langchain libraries
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import google.generativeai as genai
st.set_page_config(page_title="Player Scouting Recommendation System", page_icon="⚽", layout="wide")

# Filepath for the user database
USER_DB_FILE = "./users.json"

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load user database
def load_users():
    try:
        with open(USER_DB_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"users": []}

# Save user database
def save_users(users):
    with open(USER_DB_FILE, "w") as file:
        json.dump(users, file, indent=4)

# Registration function
def register_user(username, password):
    users = load_users()
    for user in users["users"]:
        if user["username"] == username:
            return False  # Username already exists
    users["users"].append({"username": username, "password": hash_password(password)})
    save_users(users)
    return True

# Login function
def login_user(username, password):
    users = load_users()
    for user in users["users"]:
        if user["username"] == username and user["password"] == hash_password(password):
            return True
    return False

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Login and Registration UI
if not st.session_state.logged_in:
    st.title("Login or Register")

    # Tabs for Login and Registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(f"Welcome, {login_username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Register")
        register_username = st.text_input("New Username", key="register_username")
        register_password = st.text_input("New Password", type="password", key="register_password")
        if st.button("Register"):
            if register_user(register_username, register_password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different username.")

# Main Application
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()
############### Header #################
# Set the page width to 'wide' to occupy the full width
    
    st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Scouting Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Player Scouting 2023/2024: Scout, Recommend, Elevate Your Team's Game </p>", unsafe_allow_html=True)

    ############### Simple Search Engine with Auto-Complete Query Suggestion ##############
    press = False
    choice = None

    # Initialises the streamlit session state useful for page reloading
    if 'expanded' not in st.session_state:
        st.session_state.expanded = True

    if 'choice' not in st.session_state:
        st.session_state.choice = None

    # Carica i dati dal file CSV
    df_player = pd.read_csv('football-player-stats-2023-24.csv')


    def remove_accents(text: str) -> str:
        return unidecode.unidecode(text)

    def search_csv(searchterm: str) -> List[str]:
        if searchterm:
            normalized_searchterm = remove_accents(searchterm.lower())
            df_player['NormalizedPlayer'] = df_player['Player'].apply(lambda x: remove_accents(x.lower()))
            filtered_df = df_player[df_player['NormalizedPlayer'].str.contains(normalized_searchterm, case=False, na=False)]
            suggestions = filtered_df['Player'].tolist()
            return suggestions
        else:
            return []

    selected_value = st_searchbox(
        search_csv,
        key="csv_searchbox",
        placeholder="🔍 Search a Football Player"
    )

    st.session_state.choice = selected_value
    choice = st.session_state.choice

    ################### Organic result ###########################
    if choice:
        
        # Extract column names from the JSON result
        columns_to_process = list(df_player.columns)

        # Create a normalized copy of the player DataFrame
        df_player_norm = df_player.copy()

        # Define a custom mapping for the 'Pos' column
        custom_mapping = {
            'GK': 1,
            'DF,FW': 4,
            'MF,FW': 8,
            'DF': 2,
            'DF,MF': 3,
            'MF,DF': 5,
            'MF': 6,
            'FW,DF': 7,
            'FW,MF': 9,
            'FW': 10
        }

        # Apply the custom mapping to the 'Pos' column
        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        # Select a subset of features for analysis
        selected_features = ['Pos', 'Age', 'Int',
        'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing Time MP',
        'Performance Gls', 'Performance Ast', 'Performance G+A',
        'Performance G-PK', 'Performance Fls', 'Performance Fld',
        'Performance Crs', 'Performance Recov', 'Expected xG', 'Expected npxG', 'Expected xAG',
        'Expected xA', 'Expected A-xAG', 'Expected G-xG', 'Expected np:G-xG',
        'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
        'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
        'Tackles Att 3rd', 'Challenges Att', 'Challenges Tkl%',
        'Challenges Lost', 'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
        'Standard Sh', 'Standard SoT', 'Standard SoT%', 'Standard Sh/90', 'Standard Dist', 'Standard FK',
        'Performance GA', 'Performance SoTA', 'Performance Saves',
        'Performance Save%', 'Performance CS', 'Performance CS%',
        'Penalty Kicks PKatt', 'Penalty Kicks Save%', 'SCA SCA',
        'GCA GCA', 
        'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
        'Total Cmp', 'Total Att', 'Total Cmp', 'Total TotDist',
        'Total PrgDist', '1/3'
        ]



        ####################### Cosine Similarity #######################################

        # Normalization using Min-Max scaling
        scaler = MinMaxScaler()
        df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

        # Calculate cosine similarity between players based on selected features
        similarity = cosine_similarity(df_player_norm[selected_features])

        # Find the Rk associated with the selected player's name
        index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

        # Calculate similarity scores and sort them in descending order
        similarity_score = list(enumerate(similarity[index_player]))
        similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Create a list to store data of similar players
        similar_players_data = []

        # Loop to extract information from similar players
        for player in similar_players[1:11]:  # Exclude the first player (self)
            index = player[0]
            player_records = df_player[df_player['Rk'] == index]
            if not player_records.empty:
                player_data = player_records.iloc[0]  # Get the first row (there should be only one)
                similar_players_data.append(player_data)

        # Create a DataFrame from the data of similar players
        similar_players_df = pd.DataFrame(similar_players_data)

        ########################## Analytics of the player chosen ##########################
        url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2024", limit=1, )[0]

        with st.expander("Features of The Player selected - The data considered for analysis pertains to the period of 2023 - 2024.", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(choice)
                st.image(url_player, width=356)

            with col2:
                st.caption("📄 Information of Player")
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                    st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

                with col_2:
                    st.metric("Born", df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0], None)
                    st.metric("Match Played", df_player.loc[df_player['Player'] == choice, 'Playing Time MP'].iloc[0], None, help="In 2022/2023")

                with col_3:
                    st.metric("Age", df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0], None)

                st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}", df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0], None)

            with col3:
                st.caption("⚽ Information target of Player")
                # GK
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                        col_1, col_2 = st.columns(2)

                        with col_1:
                            st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                            st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                        with col_2:
                            st.metric("Goals Against", df_player.loc[df_player['Player'] == choice, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                            st.metric("ShoTA", df_player.loc[df_player['Player'] == choice, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                # DF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                    with col_2:
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                        st.metric("Tackle", df_player.loc[df_player['Player'] == choice, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                    with col_3:
                        st.metric("Interception", df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                # MF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == choice, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                    with col_3:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                # FW
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                    col_1, col_2, col_3 = st.columns(3) 

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("xG", df_player.loc[df_player['Player'] == choice, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                        st.metric("xAG", df_player.loc[df_player['Player'] == choice, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                    with col_3:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                                
                        
        ################# Radar and Rank ######################### 
        col1, col2 = st.columns([1.2, 2])

        with col1:
            ###### Similar Players Component ###############
            st.subheader(f'Similar Players to {choice}')
            st.caption("This ranking list is determined through the application of a model based on **Cosine Similarity**. It should be noted that, being a ranking, the result obtained is inherently subjective.")
            selected_columns = ["Player", "Nation", "Squad", "Pos", "Age"]
            st.dataframe(similar_players_df[selected_columns], hide_index=True, use_container_width=True)

        with col2:
            ###### Radar Analytics #########################
            categories = ['Performance Gls', 'Performance Ast', 'KP', 'GCA GCA','Aerial Duels Won', 'Int', 'Tackles TklW', 'Performance Saves', 'Performance CS', 'Performance GA','Performance SoTA']
            selected_players = similar_players_df.head(10)

            fig = go.Figure()

            for index, player_row in selected_players.iterrows():
                player_name = player_row['Player']
                values = [player_row[col] for col in categories]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=player_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=True,  
                legend=dict(
                    orientation="v", 
                    yanchor="top",  
                    y=1,  
                    xanchor="left",  
                    x=1.02,  
                ),
                width=750,  
                height=520  
            )

            st.plotly_chart(fig, use_container_width=True)
        selected_similar_player = st.selectbox("Select a player to view stats:", similar_players_df['Player'].tolist())    
        # ...existing code...

    
        if selected_similar_player:
            st.subheader(f"Stats of {selected_similar_player}")
            player_data = df_player[df_player['Player'] == selected_similar_player]

            if not player_data.empty:
                # Get the squad name safely
                squad = player_data['Squad'].iloc[0] if 'Squad' in player_data.columns and not player_data.empty else "Unknown Team"
                
                # Construct the search query
                search_query = f"{selected_similar_player} {squad} 2023"
                
                # Get the image URL
                image_urls = bing_image_urls(search_query, limit=1)
                
                if image_urls:
                    url_player = image_urls[0]
                else:
                    url_player = "https://example.com/placeholder-image.jpg"  # Use a placeholder image URL

                with st.expander(f"Features of {selected_similar_player} - The data considered for analysis pertains to the period of 2023 - 2024.", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader(selected_similar_player)
                        st.image(url_player, width=356)

                    with col2:
                        st.caption("📄 Information of Player")
                        col_1, col_2, col_3 = st.columns(3)

                        with col_1:
                            st.metric("Nation", player_data['Nation'].iloc[0], None)
                            st.metric("Position", player_data['Pos'].iloc[0], None)

                        with col_2:
                            st.metric("Born", player_data['Born'].iloc[0], None)
                            st.metric("Match Played", player_data['Playing Time MP'].iloc[0], None, help="In 2022/2023")

                        with col_3:
                            st.metric("Age", player_data['Age'].iloc[0], None)

                        st.metric(f"🏆 League: {player_data['Comp'].iloc[0]}", player_data['Squad'].iloc[0], None)

                    with col3:
                        st.caption("⚽ Information target of Player")
                        # GK
                        if player_data['Pos'].iloc[0] == "GK":
                            col_1, col_2 = st.columns(2)

                            with col_1:
                                st.metric("Saves", player_data['Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                                st.metric("Clean Sheet", player_data['Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                            with col_2:
                                st.metric("Goals Against", player_data['Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                                st.metric("ShoTA", player_data['Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                        # DF
                        elif player_data['Pos'].iloc[0] in ["DF", "DF,MF", "DF,FW"]:
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", player_data['Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                                st.metric("Goals", player_data['Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                            with col_2:
                                st.metric("Aerial Duel", player_data['Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                                st.metric("Tackle", player_data['Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                            with col_3:
                                st.metric("Interception", player_data['Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                                st.metric("Key Passage", player_data['KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                        # MF
                        elif player_data['Pos'].iloc[0] in ["MF", "MF,DF", "MF,FW"]:
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", player_data['Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                st.metric("Goals", player_data['Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                st.metric("Aerial Duel", player_data['Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                            with col_2:
                                st.metric("GCA", player_data['GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                st.metric("Progressive PrgP", player_data['Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                            with col_3:
                                st.metric("SCA", player_data['SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                st.metric("Key Passage", player_data['KP'].iloc[0], None, help="Total number of key passes by the player.")

                        # FW
                        elif player_data['Pos'].iloc[0] in ["FW", "FW,MF", "FW,DF"]:
                            col_1, col_2, col_3 = st.columns(3) 

                            with col_1:
                                st.metric("Assist", player_data['Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                st.metric("Goals", player_data['Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                st.metric("Aerial Duel", player_data['Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                            with col_2:
                                st.metric("SCA", player_data['SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                st.metric("xG", player_data['Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                                st.metric("xAG", player_data['Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                            with col_3:
                                st.metric("GCA", player_data['GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                st.metric("Key Passage", player_data['KP'].iloc[0], None, help="Total number of key passes by the player.")

        ####################### Scouter AI Component ##################################
        
        
        dis = True
        st.header('⚽🕵️‍♂️ Scouter AI')
        message = f"Select the ideal characteristics for your team. Scouter AI will evaluate the most suitable player from the players most similar to **{choice}**"
        st.caption(message)

        api_key = st.text_input("You need to enter the Open AI API Key:", placeholder="sk-...", type="password")
        genai.configure(api_key=api_key)
        if api_key:
            dis = False

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            with st.form("my_form"):
                st.write("P R O M P T")
                # List of game styles and their descriptions
                game_styles = {
                    "Tiki-Taka": "This style of play, focuses on ball possession, control, and accurate passing.",
                    "Counter-Attack": "Teams adopting a counter-attacking style focus on solid defense and rapid advancement in attack when they regain possession of the ball.",
                    "High Press": "This style involves intense pressure on the opposing team from their half of the field. Teams practicing high pressing aim to quickly regain possession in the opponent's area, forcing mistakes under pressure.",
                    "Direct Play": "This style of play is more direct and relies on long and vertical passes, often targeting forwards or exploiting aerial play.",
                    "Pragmatic Possession": "Some teams aim to maintain ball possession as part of a defensive strategy, slowing down the game pace and limiting opponent opportunities.",
                    "Reactive": "In this style, a team adapts to the ongoing game situations, changing their tactics based on what is happening on the field. It can be used to exploit opponent weaknesses or respond to unexpected situations.",
                    "Physical and Defensive": "Some teams place greater emphasis on solid defense and physical play, aiming to frustrate opponents and limit their attacking opportunities.",
                    "Positional Play": "This style aims to dominate the midfield and create passing triangles to overcome opponent pressure. It is based on player positioning and the ability to maintain ball possession for strategic attacking.",
                    "Catenaccio": "This style, originating in Italy, focuses on defensive solidity and counterattacks. Catenaccio teams seek to minimize opponent scoring opportunities, often through zone defense and fast transition play.",
                    "Counter Attacking": "This style relies on solid defensive organization and quick transition to attack when the team regains possession of the ball. Forwards seek to exploit spaces left open by the opposing team during the defense-to-attack transition.",
                    "Long Ball": "This style involves frequent use of long and direct passes to bypass the opponent's defense. It relies on the physical strength of attackers and can be effective in aerial play situations."
                }

                # List of player experience levels
                player_experience = {
                    "Veteran": "A player with a long career and extensive experience in professional football. Often recognized for their wisdom and leadership on the field.",
                    "Experienced": "A player with experience, but not necessarily in the late stages of their career. They have solid skills and tactical knowledge acquired over time.",
                    "Young": "A player in the early or mid-career, often under 25 years old, with considerable development potential and a growing presence in professional football.",
                    "Promising": "A young talent with high potential but still needs to fully demonstrate their skills at the professional level."
                }

                # List of the leagues
                leagues = {
                    "Serie A": "Tactical and defensive football with an emphasis on defensive solidity and tactical play.",
                    "Ligue 1": "Open games with a high number of goals and a focus on discovering young talents.",
                    "Premier League": "Fast-paced, physical, and high-intensity play with a wide diversity of playing styles.",
                    "Bundesliga": "High-pressing approach and the development of young talents.",
                    "La Liga": "Possession of the ball and technical play with an emphasis on constructing actions."
                }

                # List of formations
                formations = ["4-3-1-2", "4-3-3", "3-5-2", "4-4-2", "3-4-3", "5-3-2", "4-2-3-1","4-3-2-1","3-4-1-2","3-4-2-1"]

                # List of player skills
                player_skills = [
                    "Key Passing", "Dribbling", "Speed", "Shooting", "Defending",
                    "Aerial Ability", "Tackling", "Vision", "Long Passing", "Agility", "Strength",
                    "Ball Control", "Positioning", "Finishing", "Crossing", "Marking",
                    "Work Rate", "Stamina", "Free Kicks", "Leadership","Penalty Saves","Reactiveness","Shot Stopping",
                    "Off the Ball Movement", "Teamwork", "Creativity", "Game Intelligence"
                ]

                ######### Inside FORM #####################
                st.subheader("Select a game style:")
                selected_game_style = st.selectbox("Choose a game style:", list(game_styles.keys()), disabled=dis)

                st.subheader("Select player type:")
                selected_player_experience = st.selectbox("Choose player type:", list(player_experience.keys()), disabled=dis)

                st.subheader("Select league:")
                selected_league = st.selectbox("Choose a league:", list(leagues.keys()), disabled=dis)

                st.subheader("Select formation:")
                selected_formation = st.selectbox("Choose a formation:", formations, disabled=dis)

                st.subheader("Select player skills:")
                selected_player_skills = st.multiselect("Choose player skills:", player_skills, disabled=dis)

                form = st.form_submit_button("➡️ Confirm features", disabled=dis)


        with col2:

            ######### Inside REPORT #####################
            

            if form:
                st.caption("Selected Options:")
                st.write(f"You have chosen a game style: {selected_game_style}. {game_styles[selected_game_style]} \
                This player must be {selected_player_experience} and have a good familiarity with the {selected_formation} and the skills of: {', '.join(selected_player_skills)}.")

                template = (
                    """You are a soccer scout and you must be good at finding the best talents in your team starting from the players rated by the similar player system."""
                )
                system_message_prompt = SystemMessagePromptTemplate.from_template(template)

                human_template = """
                    Generate a Football Talent Scout report based on the DATA PROVIDED (maximum 250 words) written in a formal tone FOLLOWING THE EXAMPLE.
                    It is essential to compare player attributes and select the most suitable candidate from the available options from among similar players, based on the TEAM REQUIREMENTS provided. It is important to note that the selection of players is not limited to the ranking of the players provided, as long as they meet the TEAM REQUIREMENTS.
                    THE PLAYER CHOSEN MUST NECESSARILY BE AMONG THE POSSIBLE PLAYERS CONSIDERED IN THE FOOTBALL SCOUT REPORT.
                    INDICATE the player chosen at the end of the REPORT.

                    DATA:
                    ------------------------------------
                    Similar Players List: {similar_players}
                    ------------------------------------ 

                    TEAM REQUIREMENTS:
                    Style of play: {style_t}
                    Player type required: {type_player}
                    Preferred league: {league}
                    Key ability: {ability}
                    Ideal formation: {formation}

                    EXAMPLE TO FOLLOW:
                    ### Report
                    After a detailed analysis of the data, we have identified candidates who best meet the requirements of your team. Below, we present three potential candidates:

                    ##### Three potential candidates:

                    **[Player X]**: Highlights strengths and addresses weaknesses based on data on the essential attributes for a player in his specific age group.
                    **[Player Y]**: Highlights strengths and addresses weaknesses based on data regarding the attributes a player must necessarily possess in his specific age group.
                    **[Player Z]**: Highlighting strengths and addressing weaknesses based on attribute data that a player must necessarily possess in his specific age group.
                    
                    [Provide the reasons for choosing the recommended player over the others].
                    
                    The recommended player: Name of player recommended.
                    """
                similar_players=similar_players_data
                style_t = selected_game_style
                type_player = selected_player_experience
                league = selected_league
                ability = selected_player_skills
                formation = selected_formation
                
                formatted_template = human_template.format(
                                similar_players=similar_players_data,
                                style_t=style_t,
                                type_player=type_player,
                                league=league,
                                ability=ability,
                                formation=formation
                            )
                
                human_message_prompt = HumanMessagePromptTemplate.from_template(formatted_template)

                st.caption("Text generated by Scouter AI:")
                with st.spinner("Generating text. Please wait..."):
                    generation_config = {
                            "temperature": 0.5,
                            "top_p": 0.95,
                            "top_k": 40,
                            "max_output_tokens": 8192,
                            "response_mime_type": "text/plain",
                            }
                    model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    system_instruction="You are a soccer scout and you must be good at finding the best talents in your team starting from the players rated by the similar player system",
                    
    )
                    result=model.generate_content(formatted_template)
                    

                # Extract the last item in the list
                st.markdown(result.text)

                # Use a regular expression to find the name after "The recommended player: "
                pattern = r"The recommended player:\s*([^:]+)"
                ultimo_nome = None
                # find the correspondence in the entire text
                matches = re.findall(pattern, result.text, re.IGNORECASE)
                if matches:
                    ultimo_nome = matches[0].rstrip('.')  # remove extra dot
                    if ultimo_nome.startswith('**') and ultimo_nome.endswith('**'):
                        ultimo_nome = ultimo_nome.strip('*')

        ####### Analytics of the recommended player ##############
        if form:  
            if matches:
                ultimo_nome = matches[0].rstrip('.').strip('*').strip()
        
                # Normalize player names in the DataFrame
                df_player['NormalizedPlayer'] = df_player['Player'].apply(lambda x: x.strip().lower())
                normalized_ultimo_nome = ultimo_nome.lower()
        
                # Check if the player exists in the DataFrame
                player_data = df_player[df_player['NormalizedPlayer'] == normalized_ultimo_nome]
        
                if not player_data.empty:
                    st.subheader(f"🌟 The features of the recommended player: {ultimo_nome}")
            
                    try:
                        # Get the squad name safely
                        squad = player_data['Squad'].iloc[0] if 'Squad' in player_data.columns and not player_data.empty else "Unknown Team"
                        
                        # Construct the search query
                        search_query = f"{ultimo_nome} {squad} 2023"
                        
                        # Get the image URL
                        image_urls = bing_image_urls(search_query, limit=1)
                        
                        if image_urls:
                            url_player = image_urls[0]
                        else:
                            url_player = "https://example.com/placeholder-image.jpg"  # Use a placeholder image URL
                        
                        with st.expander("Selected Player", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.subheader(ultimo_nome)
                                st.image(url_player, width=356)
                            
                            with col2:
                                st.caption("📄 Information of Player")
                                col_1, col_2, col_3 = st.columns(3)
                                
                                # Safely display player information
                                def safe_metric(label, column):
                                    value = player_data[column].iloc[0] if column in player_data.columns and not player_data.empty else "N/A"
                                    return st.metric(label, value, None)
                                
                                with col_1:
                                    safe_metric("Nation", "Nation")
                                    safe_metric("Position", "Pos")
                                
                                with col_2:
                                    safe_metric("Born", "Born")
                                    safe_metric("Match Played", "Playing Time MP")
                                
                                with col_3:
                                    safe_metric("Age", "Age")
                                
                                safe_metric(f"🏆 League: {player_data['Comp'].iloc[0] if 'Comp' in player_data.columns and not player_data.empty else 'Unknown'}", "Squad")
                            
                            # Add the code for col3 here, using the same safe_metric approach
                            with col3:
                                st.caption("⚽ Information target of Player")
                                # GK
                                if df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "GK":
                                    col_1, col_2 = st.columns(2)

                                    with col_1:
                                        st.metric("Saves", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                                        st.metric("Clean Sheet", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                                    with col_2:
                                        st.metric("Goals Against", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                                        st.metric("ShoTA", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                                # DF
                                if df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "DF,FW":
                                    col_1, col_2, col_3 = st.columns(3)

                                    with col_1:
                                        st.metric("Assist", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                                        st.metric("Goals", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                                    with col_2:
                                        st.metric("Aerial Duel", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                                        st.metric("Tackle", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                                    with col_3:
                                        st.metric("Interception", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                                        st.metric("Key Passage", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                                # MF
                                if df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "MF,FW":
                                    col_1, col_2, col_3 = st.columns(3)

                                    with col_1:
                                        st.metric("Assist", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                        st.metric("Goals", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                        st.metric("Aerial Duel", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                                    with col_2:
                                        st.metric("GCA", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                        st.metric("Progressive PrgP", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                                    with col_3:
                                        st.metric("SCA", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                        st.metric("Key Passage", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                                # FW
                                if df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Pos'].iloc[0] == "FW,DF":
                                    col_1, col_2, col_3 = st.columns(3) 

                                    with col_1:
                                        st.metric("Assist", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                        st.metric("Goals", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                        st.metric("Aerial Duel", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                                    with col_2:
                                        st.metric("SCA", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                        st.metric("xG", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                                        st.metric("xAG", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                                    with col_3:
                                        st.metric("GCA", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                        st.metric("Key Passage", df_player.loc[df_player['NormalizedPlayer'] == normalized_ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")
                    except Exception as e:
                        st.error(f"An error occurred while processing player data: {str(e)}")
                else:
                    st.error(f"Player '{ultimo_nome}' not found in the database. Please ensure the AI recommends players from the provided similar players list.")
                    st.write("Available players:", ", ".join(similar_players_df['Player'].tolist()))
            else:
                st.warning("No player recommendation found in the AI's response.")                                        


        st.write(" ")
else:
    st.warning("You must log in to access the Player Scouting Recommendation System.")
  
