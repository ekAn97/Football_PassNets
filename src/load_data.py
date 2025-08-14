import pandas as pd
import numpy as np
import datetime
from mplsoccer import Pitch, Sbopen

'''
In this script we store functions that load match and event data from the statsbomb database.
We also define functions that obtain the passing data for different phases of play and filter the data
to get passes of interest (e.g. forward passes, build-up passes, etc.). Additional helping functions for
mathematical calculations are also included.
'''

#### HELP FUNCTIONS FOR CALCULATIONS ####
def cosine_pass_vector(pass_data):
    '''
    A function that calculates the cosine of the angle of a pass

    INPUT:
        pass_data: pd.DataFrame with passing data

    OUTPUT:
        pass_data: pd.DataFrame with a "cosine" column
    '''
    # Get x, y coordinates
    x_start, y_start = np.array(pass_data["x"].to_list()), np.array(pass_data["y"].to_list())
    x_end, y_end = np.array(pass_data["end_x"].to_list()), np.array(pass_data["end_y"].to_list())

    delta_x = x_end - x_start
    delta_y = y_end - y_start

    # Euclidean norm
    norm = np.hypot(delta_x, delta_y)

    # Create the cosine column in the pass data frame
    pass_data["cosine"] = np.divide(delta_x, norm)

    return pass_data 


#### LOAD AND OBTAIN DESIRED DATA FUNCTIONS ####
def get_match_data(competition, season):
    '''
    Open data specified in the configuration input file. Get the matches
    for the specified competition and season

    INPUT
        competition: str, competition name
        season: str, season name

    OUTPUT
        parser: parser object to retrieve data
        match_data: pd.DataFrame with the competition's match data
    '''
    # Open Data
    parser = Sbopen(dataframe = True)
    # Get Competition Info
    competitions = parser.competition()
    # Get Indices and Match Data for Specific Competition
    competition_df = competitions[(competitions["competition_name"] == competition) & (competitions["season_name"] == season)]
    competition_id, season_id = competition_df["competition_id"].values[0], competition_df["season_id"].values[0]
    match_data = parser.match(competition_id = competition_id, season_id = season_id)

    return parser, match_data

def get_match_info(parser, match_data, match_id):
    '''
    Get general match info like final score, line-ups, match date etc. to 
    use for illustration purposes in the Streamlit app

    INPUT:
        parser object for statsbomb data
        match_data: pd.DataFrame with match info
        match_id: int, the identifier for a specific match

    OUTPUT:
        match_info_dict: dict with important match info
    '''
    match_data = match_data[match_data["match_id"] == match_id]
    match_info_dict = {
        "Home Team": match_data["home_team_name"].values[0],
        "Away Team": match_data["away_team_name"].values[0],
        "Final Score": str(match_data["home_score"].values[0]) + '-' + str(match_data["away_score"].values[0]),
        "Match Date": match_data["match_date"].values[0],
        "Stage": match_data["competition_stage_name"].values[0],
        "Stadium": match_data["stadium_name"].values[0],
    }
    transf_date = datetime.datetime.utcfromtimestamp(match_info_dict["Match Date"].astype('datetime64[s]').astype(int))

    # Format to friendly string
    match_info_dict["Match Date"] = transf_date.strftime("%B %d, %Y")

    # Lineup & jersey number info
    lineups = parser.lineup(match_id)

    return match_info_dict, lineups

def phase_of_play(event_data, team_name):
    '''
    Define phases of play according to substitutions that happened during the game

    INPUT
        event_data: pd.DataFrame with event data
        team_name: str

    OUTPUT
        idx_bound: list, contains index boundaries that signal the start and end of each phase

    '''
    subs_id = event_data[(event_data["type_name"] == "Substitution") & (event_data["team_name"] == team_name)]["index"].values
    
    if len(subs_id) == 0: # case of zero subs
        return []
    
    # Group consecutive substitutions
    idx_bound = []
    current_idx_group = [subs_id[0]]

    for _ in range(1, len(subs_id)):
        if subs_id[_] == subs_id[_ - 1] + 1: # consecutive subs
            current_idx_group.append(subs_id[_])
        else: # non-consecutive subs
            idx_bound.append(max(current_idx_group))
            current_idx_group = [subs_id[_]]
    
    idx_bound.append(max(current_idx_group))

    return idx_bound


def get_passing_data(event_data, lineup, team_name, phase_idx, subs_id):
    '''
    Creating the pass data frame from the event data for the selected phase

    INPUT
        event_data: pd.DataFrame with event data
        lineup: pd.DataFrame with lineups to obtain jersey numbers
        team_name: str
        phase_idx: int, index that corresponds to the phase selected by user
        subs_id: array, array of indices corresponding to substitution events

    OUTPUT
        pass_data: pd.DataFrame with x,y positions of passer and receiver
    '''
    if len(subs_id) == 0: # case of zero substitutions
        start_idx, end_idx = 0, len(event_data)
    else:
        start_idx = 0 if phase_idx == 0 else subs_id[phase_idx - 1]
        end_idx = subs_id[phase_idx] if phase_idx < len(subs_id) else len(event_data)

    mask = (
        (event_data.type_name == "Pass") &
        (event_data.team_name == team_name) &
        (event_data.outcome_name.isnull()) &
        (event_data.sub_type_name != "Throw-in")
    )

    pass_data = event_data.loc[start_idx: end_idx, :].loc[mask, ["x", "y", "end_x", "end_y", "pass_length", "player_id", "player_name", "pass_recipient_id", "pass_recipient_name"]]

    # Transform pass length from yards to meters (1 yard = 0.9144 meters)
    pass_data["pass_length"] = pass_data["pass_length"].apply(lambda x: x * 0.9144)

    # Keep player surname only
    pass_data["player_name"] = pass_data["player_name"].apply(lambda x: str(x).split()[-1])
    pass_data["pass_recipient_name"] = pass_data["pass_recipient_name"].apply(lambda x: str(x).split()[-1])
    
    # Assign jersey number to passer according to lineups data frame
    pass_data = pass_data.merge(
        lineup[["player_id", "jersey_number"]],
        on = "player_id",
        how = "left"
    ).rename(columns = {"jersey_number": "player_name_jersey"})

    # Assign jersey number to recipient according to lineups data frame
    pass_data = pass_data.merge(
        lineup[["player_id", "jersey_number"]],
        left_on = "pass_recipient_id",
        right_on = "player_id",
        how = "left",
        suffixes = ("", "_recipient")
    ).rename(columns = {"jersey_number": "pass_recipient_jersey"})

    pass_data = pass_data.drop(columns = ["player_id_recipient"])
    pass_data["player_name_jersey"] = pass_data["player_name_jersey"].apply(lambda x: str(x))
    pass_data["pass_recipient_jersey"] = pass_data["pass_recipient_jersey"].apply(lambda x: str(x))

    return pass_data


def region_pass_filter(pass_data, config, pitch_type, pitch_third):
    '''
    Divide the pitch in three third: defensive, midfield and attacking thirds.
    Query the passing data accordingly to get the passes that occurred in the
    desired third.

    INPUT:
        pass_data: pd.DataFrame, all pass events except throw-ins
        config: json/dict, import the parsed config file to get the pitch name and dimensions
        pitch_type: str, name of data provider
        pitch_third: str, "def", "mid" or "att"

    OUTPUT:
        first_bound: float, signals the end of the defensive third
        second_bound: float, signals the end of the midfield third
        filtered_pass_data: pd.DataFrame, contains pass events in the specified third 
    '''
    # Define the pitch and set the coordinates
    pitch_info = config["pitch_type"][pitch_type]
    pitch = Pitch(pitch_type = pitch_type, axis = False, label = False)
    length, width = pitch_info["pitch_length"], pitch_info["pitch_width"]
    first_bound, second_bound = np.round((length / 3), 2), np.round((2 * length) / 3, 2)

    if pitch_third == "def": # defensive third
        x_start, x_end = 0, first_bound
        filtered_pass_data = pass_data[(pass_data["x"] >= x_start) & (pass_data["x"] < x_end)]

    elif pitch_third == "mid": # midfield third
        x_start, x_end = first_bound, second_bound
        filtered_pass_data = pass_data[(pass_data["x"] >= x_start) & (pass_data["x"] < x_end)]

    elif pitch_third == "att": # attacking third
        x_start, x_end = second_bound, length
        filtered_pass_data = pass_data[(pass_data["x"] >= x_start) & (pass_data["x"] <= x_end)]

    return first_bound, second_bound, filtered_pass_data


def direction_pass_filter(pass_data, direction):
    '''
    Filter passes according to direction to obtain forward, backward or lateral passes. All three categories
    adhere to the definition provided by WyScout: https://dataglossary.wyscout.com/pass/

    INPUT
        pass_data: pd.DataFrame, can be either the overall pass events or the pass events within a specific region
        direction: str, "back", "fwd", "lat"

    OUTPUT
        filtered_passes: pd.DataFrame, filtered data according to direction
    '''
    # Calculate the boundaries for classifying passes (counter clock-wise starting from 0 degrees)
    fwd_bound = np.cos(np.pi / 4)

    # Filter passes
    if direction == "back": # backward passes
        filtered_passes = pass_data[pass_data["cosine"] < - fwd_bound]

    elif direction == "lat": # lateral passes
        filtered_passes = pass_data[(pass_data["cosine"] >= - fwd_bound) & (pass_data["cosine"] <= fwd_bound)]
        filtered_passes = filtered_passes[filtered_passes["pass_length"] > 12]

    elif direction == "fwd": # forward passes; we retrieve two data frames
        filtered_passes = pass_data[pass_data["cosine"] > fwd_bound]

    return filtered_passes

def progressive_passes(pass_data, region_change, first_bound, second_bound):
    '''
    Obtain the progressive passes, i.e. passes that move the ball from the one third to the other.

    INPUT:
        pass_data: pd.DataFrame, event passing data stored in a dataframe format (overall forward passes)
        region_change: str, either "d2m" (defensive to midfield third) or "m2a" (midfield to attacking third) or "d2a" (defensive to attacking third)
        first_bound, second_bound: float, denote the bounds for the three pitch regions
    
    OUTPUT:
        pass_data: pd.DataFrame, with filtered passes
    '''
    if region_change == "d2m": # Progress ball from defensive third to the midfield third
        pass_data = pass_data[(pass_data["x"] < first_bound) & (pass_data["end_x"] >= first_bound)\
                              & (pass_data["end_x"] < second_bound)]
        
    elif region_change == "m2a": # Progress ball from midfield third to the attacking third
        pass_data = pass_data[(pass_data["x"] >= first_bound) & (pass_data["x"] < second_bound)\
                              & (pass_data["end_x"] >= second_bound)]
    
    elif region_change == "d2m": # Progress ball from defensive third to the attacking third
        pass_data = pass_data[(pass_data["x"] < first_bound) & (pass_data["end_x"] >= second_bound)]

    return pass_data