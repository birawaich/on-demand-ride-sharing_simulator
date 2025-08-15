import osmnx as ox
import networkx as nx

highway_filt = '["highway"~"primary|secondary|tertiary|residential|unclassified|road|living_street"]'
# construction_filt = '["construction"~"unclassified|residential|living_street|road"]'
# total_filt = [highway_filt, construction_filt]
G = ox.graph_from_place('Manhattan', simplify=True,custom_filter=highway_filt)
                        #  custom_filter=[“highway”~"primary|secondary|tertiary|residential|unclassified|road|living street"])
print("Graph downloaded successfully.")

ig, ax = ox.plot.plot_graph(
    G,
    ax=None,  # optionally draw on pre-existing axis
    figsize=(8, 8),  # figure size to create if ax is None
    bgcolor="#111111",  # background color of the plot
    node_color="w",  # color of the nodes
    node_size=15,  # size of the nodes: if 0, skip plotting them
    node_alpha=None,  # opacity of the nodes
    node_edgecolor="none",  # color of the nodes' markers' borders
    node_zorder=1,  # zorder to plot nodes: edges are always 1
    edge_color="#999999",  # color of the edges
    edge_linewidth=1,  # width of the edges: if 0, skip plotting them
    edge_alpha=None,  # opacity of the edges
    show=True,  # if True, call pyplot.show() to show the figure
    close=False,  # if True, call pyplot.close() to close the figure
    save=False,  # if True, save figure to disk at filepath
    filepath=None,  # if save is True, the path to the file
    dpi=300,  # if save is True, the resolution of saved file
    bbox=None,  # bounding box to constrain plot
)

print(nx.is_strongly_connected(G))
ox.io.save_graphml(G, filepath='man_graph.graphml')



