import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from load_data import get_match_data, get_match_info, phase_of_play

st.set_page_config(page_title = "Passing Network Analysis Toolkit", layout = "wide")


############################################################################################################################
#                                                                                                                          #
#                                                   L O A D  D A T A                                                       #
#                                                                                                                          #
############################################################################################################################

# Read config.json file to fetch competition and season name
with open("config.json", "r") as file:
    params = json.load(file)

# Get competition match data (get_match_data)
parser, match_data = get_match_data(**params) # len(match_data) = number of matches

@st.cache_data
def load_match_info(_parser, match_df, match_id):
    return get_match_info(_parser, match_df, match_id)

@st.cache_data
def load_events(_parser, match_id):
    return _parser.event(match_id)

st.sidebar.title("Match & Team selection")
# Set selection boxes ( Select competition round)
comp_round = st.sidebar.selectbox(
    "Select competition round",
    ["--Select--", "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
)
fixture = "--Select--"
team_analyzed = None

if comp_round != "--Select--":
    # Filter matches for the selected round & build fixture list for selection
    comp_round_matches = match_data[match_data["competition_stage_name"] == comp_round]
    comp_round_matches["fixture"] = comp_round_matches["home_team_name"] + " vs " + comp_round_matches["away_team_name"]

    fixtures = comp_round_matches.sort_values("match_date")["fixture"].tolist()

    fixture = st.sidebar.selectbox(
        "Select fixture",
        ["--Select--"] + fixtures,
        help = "Fixtures are sorted by date"
    )

    if fixture != "--Select--":
        # Get match_id for the selected fixture & obtain match event data
        match_id = comp_round_matches[comp_round_matches["fixture"] == fixture]["match_id"].iloc[0]
        match_info, lineups = load_match_info(parser, comp_round_matches, match_id)
        event_df, related, freeze, tactics = load_events(parser, match_id)
        
        # Select team
        teams = comp_round_matches[comp_round_matches["fixture"] == fixture][["home_team_name", "away_team_name"]].iloc[0].tolist()
        team_analyzed = st.sidebar.radio(
            "Select team for analysis",
            teams
        )
        
####################################################################################################################
#                                                                                                                  #
#                                                       M A I N                                                    #
#                                                                                                                  #
####################################################################################################################

        # Get substitution events indices to generate phases of play for the selection box
def display_match_header(match_info):
    st.markdown(
        f"""
        <h2 style='text-align: center;'>{match_info["Home Team"]} {match_info["Final Score"]} {match_info["Away Team"]}</h2>
        <p style='text-align: center; font-size:16px; color:gray;'>
        📅 {match_info["Match Date"]}, 📍 {match_info["Stadium"]}
        </p>
        """,
        unsafe_allow_html=True
        )
    
if fixture != "--Select--" and team_analyzed:
    # Extract lineup for selected team
    lineup_df = lineups[lineups["team_name"] == team_analyzed][["player_id", "player_name", "jersey_number"]].copy()
    lineup_df["surname"] = lineup_df["player_name"].apply(lambda x: x.split()[-1])

    display_lineup = lineup_df[["jersey_number", "surname"]].sort_values("jersey_number")
    display_lineup = display_lineup.rename(columns={"jersey_number": "#", "surname": "Surname"})

    html_df = display_lineup.to_html(index=False, classes="table", border=0)

    num_players = st.sidebar.slider(
        "Number of players to show",
        min_value=1,
        max_value=len(display_lineup),
        value=len(display_lineup)
    )

    # Filter lineup based on slider
    display_lineup_subset = display_lineup.head(num_players)
    display_lineup_subset = display_lineup_subset.to_html(index=False, classes="table", border=0)

    st.sidebar.markdown(f"### {team_analyzed} Lineup")
    st.sidebar.markdown(display_lineup_subset, unsafe_allow_html=True)

    # Display match header
    display_match_header(match_info)

    # Select phase of play
    subs_id = phase_of_play(event_df, team_analyzed)
    phases = [f"Phase {_}" for _ in range(1, len(subs_id) + 1)] # phases of play are now determined by time instead of number of changes
    phase_selected = st.radio(
        "Select phase of play",
        phases,
        help = "A phase of play ends right after a substitution happens.",
        horizontal = True
    )
    phase_idx = int(phase_selected.split(" ")[1]) - 1 # create a dummy index for easier querying
    
    # Tabs set up
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Passing Networks"

    tabs = st.tabs(["Passing Networks", "Team Stats", "Individual Stats"])
    tab_names = ["Passing Networks", "Team Stats", "Individual Stats"]

    for i, tab in enumerate(tabs):
        with tab:
            if tab_names[i] == "Passing Networks":
                st.subheader(f"Passing Network – {team_analyzed}")
                st.write("⚡ Overall Passing network, build-up network and final third network visualizations")
            elif tab_names[i] == "Team Stats":
                st.subheader("Team-Level Metrics")
                st.write("📊 Gini bar plots across phases with average centralization and top player combinations table go here.")
            elif tab_names[i] == "Player Stats":
                st.subheader("Player-Level Metrics")
                st.write("📋 Sortable player stats table with build-up/final-third counts go here.")
else:
     st.markdown("<p style='text-align:center; color:gray;'>Select a round, fixture, and team to start the analysis.</p>", unsafe_allow_html=True)
        