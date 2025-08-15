import osmnx as ox
import igraph as ig
import networkx as nx
import uuid
import operator
import numpy as np
# Use the following commands to download the graph file of NYC
G = ox.io.load_graphml('code/simulation/man_graph.graphml')
# G = ox.io.load_graphml('man_graph.graphml')
weight = "travel_time"
print("Graph downloaded successfully.")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
def get_largest_scc(G):
    """
    Extract the largest strongly connected component.
    """
    # Find all strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    
    # Get the largest one
    largest_scc = max(sccs, key=len)
    
    # Create subgraph with only the largest SCC
    G_scc = G.subgraph(largest_scc).copy()
    
    return G_scc

G = get_largest_scc(G)
G = nx.convert_node_labels_to_integers(G)
osmids = list(G.nodes)
osmid_values = dict(zip(G.nodes, osmids))
nx.set_node_attributes(G, osmid_values, "osmid")

locations: dict[uuid.UUID,tuple] = {}
locations_list = []
for node in G.nodes:
    loc_uuid = uuid.uuid4()
    loc_x = G.nodes[node]['x'] #convert to grid coordinates
    loc_y = G.nodes[node]['y'] #convert to grid coordinates
    locations_list.append(loc_uuid)
    locations[loc_uuid] = (loc_x, loc_y)

G_ig = ig.Graph(directed=True)
G_ig = ig.Graph(directed=True)
G_ig.add_vertices(G.nodes)
G_ig.add_edges(G.edges())
G_ig.vs["osmid"] = osmids
G_ig.es[weight] = list(nx.get_edge_attributes(G, weight).values())



source = next(iter(G.nodes()))
target = list(G.nodes())[-3]
target2 = list(G.nodes())[-1]

path1 = G_ig.get_shortest_paths(v=source, to=target, weights=weight)[0]
path2 = G_ig.get_shortest_paths(v=target, to=target2, weights=weight)[0]
gdf1 = ox.routing.route_to_gdf(G,path1,weight=weight)
time1 = gdf1['travel_time'].sum()
times = gdf1['travel_time'].values
tmp = operator.itemgetter(*path2)(locations_list)
minx = min(G.nodes[node]['x'] for node in G.nodes)
maxx = max(G.nodes[node]['x'] for node in G.nodes)
miny = min(G.nodes[node]['y'] for node in G.nodes)
maxy = max(G.nodes[node]['y'] for node in G.nodes)

color_road = "#d4d4d4ff"
width_road = 2
# fig, ax = ox.plot.plot_graph(
#   G,
#   ax=None,  # optionally draw on pre-existing axis
#   figsize=(8, 8),  # figure size to create if ax is None
#   bgcolor="w",  # background color of the plot
#   node_color="g",  # color of the nodes
#   node_size=10,  # size of the nodes: if 0, skip plotting them
#   node_alpha=None,  # opacity of the nodes
#   node_edgecolor="none",  # color of the nodes' markers' borders
#   node_zorder=1,  # zorder to plot nodes: edges are always 1
#   edge_color=color_road,  # color of the edges
#   edge_linewidth=width_road,  # width of the edges: if 0, skip plotting them
#   edge_alpha=None,  # opacity of the edges
#   show=True,  # if True, call pyplot.show() to show the figure
#   close=False,  # if True, call pyplot.close() to close the figure
#   save=False,  # if True, save figure to disk at filepath
#   filepath=None,  # if save is True, the path to the file
#   dpi=700,  # if save is True, the resolution of saved file
#   bbox=None,  # bounding box to constrain plot
# )

# while not nx.is_strongly_connected(G):
#     for i in range(0, len(G.nodes)):
#         print(f"Checking node {i} of {len(G.nodes)}")
#         for j in range(1, len(G.nodes)):
#             if i == j:
#                 continue
#             path = G.get_shortest_path(source=i, target=j, weight=weight)
#             if len(path) == 0 and j not in badids:
#                 badids.append(j)

distances = np.zeros((len(G.nodes), len(G.nodes)))
for i, source in enumerate(G.nodes):
    print(f"Calculating distances from node {i} of {len(G.nodes)}")
    for j, target in enumerate(G.nodes):
        if source == target:
            distances[i][j] = 0
            continue
        path1 = G_ig.get_shortest_paths(v=source, to=target, weights=weight)[0]
        gdf1 = ox.routing.route_to_gdf(G,path1,weight=weight)
        distances[i][j] = gdf1['travel_time'].sum()
np.save('distances.npy', distances)
print(nx.is_strongly_connected(G))
ox.io.save_graphml(G, filepath='man_graph_prune.graphml')
fig, ax = ox.plot.plot_graph(
  G,
  ax=None,  # optionally draw on pre-existing axis
  figsize=(8, 8),  # figure size to create if ax is None
  bgcolor="w",  # background color of the plot
  node_color="g",  # color of the nodes
  node_size=10,  # size of the nodes: if 0, skip plotting them
  node_alpha=None,  # opacity of the nodes
  node_edgecolor="none",  # color of the nodes' markers' borders
  node_zorder=1,  # zorder to plot nodes: edges are always 1
  edge_color=color_road,  # color of the edges
  edge_linewidth=width_road,  # width of the edges: if 0, skip plotting them
  edge_alpha=None,  # opacity of the edges
  show=True,  # if True, call pyplot.show() to show the figure
  close=False,  # if True, call pyplot.close() to close the figure
  save=False,  # if True, save figure to disk at filepath
  filepath=None,  # if save is True, the path to the file
  dpi=700,  # if save is True, the resolution of saved file
  bbox=None,  # bounding box to constrain plot
)
