import pandas as pd
import networkx as nx
import numpy as np
from typing import Literal


def create_graph(pass_data):
    '''
    Create the corresponding graph from the pass event data frame.

    INPUT:
        pass_data: pd.DataFrame, pass event data
    
    OUTPUT:
        G: nx.MultiDiGraph, graph object with edgelist, edge attributes and node attributes
    '''
    # Create an edgelist from the pass event data
    pass_net_edgelist = (
        pass_data.groupby(["player_name_jersey", "pass_recipient_jersey"])
        .size()
        .reset_index(name = "intensity")
        .rename(columns = {"player_name_jersey": "passer", "pass_recipient_jersey": "receiver"})
    )

    # Create a DiGraph object
    G = nx.from_pandas_edgelist(
        pass_net_edgelist,
        source = "passer",
        target = "receiver",
        edge_attr = "intensity",
        create_using = nx.MultiDiGraph
    )

    # Set average position as nodal attribute
    avg_pos = {}
    for player in G.nodes():
        # Fetch x, y, endx, endy positions of player
        player_x = pass_data[pass_data["player_name_jersey"] == player]["x"].to_numpy()
        player_y = pass_data[pass_data["player_name_jersey"] == player]["y"].to_numpy()
        player_endx = pass_data[pass_data["pass_recipient_jersey"] == player]["end_x"].to_numpy()
        player_endy = pass_data[pass_data["pass_recipient_jersey"] == player]["end_y"].to_numpy()

        avg_Xpos = float(np.round(np.mean(np.concatenate([player_x, player_endx])), 2))
        avg_Ypos = float(np.round(np.mean(np.concatenate([player_y, player_endy])), 2))

        avg_pos.update({player: [avg_Xpos, avg_Ypos]})

    nx.set_node_attributes(G, avg_pos, "avg_pos")

    # Set edge attribute: Distance (defined as the inverse of edge weights, i.e. passes)
    distances = {}
    for key, values in nx.get_edge_attributes(G, "intensity").items():
        distances.update({key: round(1 / values, 4) * 10000})

    nx.set_edge_attributes(G, distances, name = "distance")

    return G

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
            name = "in-strength"
        elif direction == "out":
            strength_values[node] = int(sum(adj_matrix[idx]))
            name = "out-strength"
        else:
            strength_values[node] = int(sum(adj_matrix[:, idx]) + sum(adj_matrix[idx]))
            name = "strength"
            
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
    if centrality_name == "betweenness":
        centrality_dict = nx.betweenness_centrality(G, normalized = True, weight = distance_attribute)
    elif centrality_name == "in-harmonic":
        centrality_dict = nx.harmonic_centrality(G, distance = distance_attribute)
    elif centrality_name == "out-harmonic":
        Grev = G.reverse(copy = True)
        centrality_dict = nx.harmonic_centrality(Grev, distance = distance_attribute)

    # Up to four (4) decimals
    for key, value in centrality_dict.items():
        centrality_dict[key] = round(value, 4)
        
    # Store strength values as node attribute
    nx.set_node_attributes(G, centrality_dict, name = centrality_name)