def get_passing_data(event_data, team_name, sub_idx):
    '''
    Creating the pass data frame from the event data

    INPUT
        event_data: pd.DataFrame with event data
        team_name: str
        sub_idx: int, index that corresponds to the substitution event

    OUTPUT
        pass_data: pd.DataFrame with x,y positions of passer and receiver
        pass_network_df: pd.DataFrame to be used as an edgelist for Graph creation
    '''
    pass_data = (event_data.type_name == "Pass") & (event_data.team_name == team_name) & (event_data.index < sub_idx) & (event_data.outcome_name.isnull()) & (event_data.sub_type_name != "Throw-in")
    pass_data = event_data.loc[pass_data, ["x", "y", "end_x", "end_y", "player_name", "pass_recipient_name"]]
    # Keep player surname only and jersey number
    pass_data["player_name"] = pass_data["player_name"].apply(lambda x: str(x).split()[-1])
    pass_data["pass_recipient_name"] = pass_data["pass_recipient_name"].apply(lambda x: str(x).split()[-1])
    # Passing network data frame
    pass_net_edgelist = (
        pass_data.groupby(["player_name", "pass_recipient_name"])
        .size()
        .reset_index(name = "Intensity")
        .rename(columns = {"player_name": "Passer", "pass_recipient_name": "Receiver"})
    )

    # Create a DiGraph object
    G = nx.from_pandas_edgelist(
        pass_net_edgelist,
        source = "Passer",
        target = "Receiver",
        edge_attr = "Intensity",
        create_using = nx.MultiDiGraph
    )

    # Set average position as nodal attribute
    avg_pos = {}
    for player in G.nodes():
        # Get x, y, endx, endy positions of player only
        player_x = pass_data[pass_data["player_name"] == player]["x"].to_numpy()
        player_y = pass_data[pass_data["player_name"] == player]["y"].to_numpy()
        player_endx = pass_data[pass_data["pass_recipient_name"] == player]["end_x"].to_numpy()
        player_endy = pass_data[pass_data["pass_recipient_name"] == player]["end_y"].to_numpy()
        
        avg_Xpos = float(np.round(np.mean(np.concatenate([player_x, player_endx])), 2))
        avg_Ypos = float(np.round(np.mean(np.concatenate([player_y, player_endy])), 2))

        avg_pos.update({player: [avg_Xpos, avg_Ypos]})

    # Set node attribute: average X and average Y position
    nx.set_node_attributes(G, avg_pos, "Avg Pos")

    # Set edge attribute: Distance defined as the inverse of edge weights, i.e. passes
    distances = {}
    for key, values in nx.get_edge_attributes(G, "Intensity").items():
        distances.update({key: round(1 / values, 4) * 10000})

    nx.set_edge_attributes(G, distances, name = "Distance")
    
    return pass_data, pass_net_edgelist, G

def node_strength(G, direction: Literal["in", "out", None], weight):
    '''
    Calculates the strength (weighted degree) of the nodes in a graph for
    a directed and undirected network.

    INPUT:
        G: networkx graph object
        direction:  "in", "out" or None; specifies the directionality of the network
        weight: str, denotes the variable that acts as an edge weight
        
    OUTPUT:
        -
    '''
    # Get the adjacency matrix and create a dictionary to store strength values
    adj_matrix = nx.to_numpy_array(G, weight = weight)
    strength_values = {node: None for node in G.nodes()}

    # Calculation of strength according to direction. Adjacency matrix is read 
    # from Left to Right
    for idx, node in enumerate(G.nodes()):
        if direction == "in":
            strength_values[node] = int(sum(adj_matrix[:, idx]))
            name = "In-Strength"
        elif direction == "out":
            strength_values[node] = int(sum(adj_matrix[idx]))
            name = "Out-Strength"
        else:
            strength_values[node] = int(sum(adj_matrix[:, idx]) + sum(adj_matrix[idx]))
            name = "Strength"
            
    # Store strength values as node attribute
    nx.set_node_attributes(G, strength_values, name = name)

def distance_centralities(G, centrality_name, distance_attribute):
    '''
    Calculate distance-related centralities like Betweenness, Harmonic Closeness

    INPUT:
        G: networkx Graph object
        centrality_name: str
        distance_attribute: str, specify the name of the edge attribute used as distance

    OUTPUT:

    '''
    if centrality_name == "Betweenness":
        centrality_dict = nx.betweenness_centrality(G, normalized = True, weight = distance_attribute)
    elif centrality_name == "In-Harmonic":
        centrality_dict = nx.harmonic_centrality(G, distance = distance_attribute)
    elif centrality_name == "Out-Harmonic":
        Grev = G.reverse(copy = True)
        centrality_dict = nx.harmonic_centrality(Grev, distance = distance_attribute)

    # Up to four (4) decimals
    for key, value in centrality_dict.items():
        centrality_dict[key] = round(value, 4)
        
    # Store strength values as node attribute
    nx.set_node_attributes(G, centrality_dict, name = centrality_name)