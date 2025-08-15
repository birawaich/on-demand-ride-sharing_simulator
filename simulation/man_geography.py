import uuid
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import warnings
import copy
import operator
import osmnx as ox
import networkx as nx
import igraph as ig
from queue import PriorityQueue
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, BoxStyle
from geopy.distance import geodesic

from visualization import VehicleMarker
from geo import Geography
from simetime import SimDuration

import os


@dataclass
class PointInformation():
    """PRIVATE dataclass holding information about one point"""
    id: uuid.UUID
    coordinate_x: int #in coordinate points i.e. km
    coordinate_y: int #in coordinate points i.e. km
    coordinate_vector: np.array #in m
    name: str
    #Scales poorly with larger grid sizes (O(N^2) in memory?), but this is probably more efficient than computing distances at runtimes
    dist_to_nodes: dict[uuid.UUID, int] = field(default_factory=dict)  # distance from each node to other nodes determined by their UUID 
    # To access the distance, use locations[current_node_uuid].dist_to_node[other_node_uuid]
    

class ManGridGeography(Geography):
    """
    Geography class for Bielegrid

    BieleGrid = nxn grid with gridpoints some fixed distance appart
    Vehicles have a fixed speed, no traffic is present i.e. travel times are not dependent on time of the day.

    Attributes
    ----------
    _locations: dict[uuid.UUID,PointInformation]
        (private) Dictionary that contains all information about the different location points
    _grid_sesperation: float
        (private) seperation of grid in m, defaults to 1km
    _vehicle_speed_mps: float
        (private) vehicle speed in m/s, defaults to 50km/h
    """


    DEBUG_PRINT = False
    """Class Property: set to true for verbose prints"""
    def __init__(self, precompute_dist: bool=True):
       
        # Map of location ids to index in the and distance matrix
        self._loc_list: list[uuid.UUID] = []
        self._loc_map: dict[uuid.UUID,int] = {} #Map of index to location id
        self.PRECOMPUTE = precompute_dist #whether to precompute distances or not
        # Load Grid 
        dir_path = os.path.dirname(os.path.realpath(__file__)) #get current directory

        G = ox.io.load_graphml(dir_path+'/man_graph_prune.graphml')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = nx.convert_node_labels_to_integers(G)
        locations: dict[uuid.UUID,PointInformation] = {}
        for node in G.nodes:
            loc_uuid = uuid.uuid4()
            self._loc_list.append(loc_uuid) #add to map
            self._loc_map.update({loc_uuid: len(self._loc_list)-1}) #add to map
            loc_x = G.nodes[node]['x'] #convert to grid coordinates
            loc_y = G.nodes[node]['y'] #convert to grid coordinates
            locations.update({loc_uuid: #uuid as key,)
                                 PointInformation( #point information as value
                                     id=loc_uuid,
                                     coordinate_x=loc_x,
                                     coordinate_y=loc_y,
                                     coordinate_vector=np.array([loc_x, loc_y], dtype=float),
                                     name=f"({loc_x},{loc_y}) {loc_uuid}")})
        osmids = list(G.nodes)
        self.n_nodes = len(G.nodes)
        self._distance_matrix: np.ndarray = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float16)
        osmid_values = dict(zip(G.nodes, osmids))
        nx.set_node_attributes(G, osmid_values, "osmid")
        G_ig = ig.Graph(directed=True)
        G_ig = ig.Graph(directed=True)
        G_ig.add_vertices(G.nodes)
        G_ig.add_edges(G.edges())
        self.weight = "travel_time"
        G_ig.vs["osmid"] = osmids
        G_ig.es[self.weight] = list(nx.get_edge_attributes(G, self.weight).values())
        self.G:nx.classes.multidigraph = G
        self.G_ig:ig.Graph = G_ig
      
        self._locations: dict[uuid.UUID,PointInformation] = locations #location lookup table
        # compute distance matrix
        if self.PRECOMPUTE:
          self._compute_distance_matrix() #NOTE DISTANCE MATRIX IS MEASURING TIME BETWEEN LOCATIONS, NOT DISTANCE

    ### PUBLIC

    def get_location_ids(self) -> set[uuid.UUID]:
      """
      Return the location IDs as a set
      """
      return set(self._locations.keys())
    
    def predict_travel_duration(self,
                                departure_location: uuid.UUID,
                                arrival_location: uuid.UUID) -> SimDuration:
      travel_time = self.get_distance(departure_location, arrival_location)
      return SimDuration(travel_time) # Convert to SimDuration
    
    def get_location_coordinates(self,
                                 location: uuid.UUID,
                                 do_assert_location = True) -> np.ndarray:
        if do_assert_location:
            assert location in self._locations, \
                "Location is not valid!"

        return copy.copy(self._locations[location].coordinate_vector)
    
    def get_limits_location_coordinates(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Return the minimal and maximal coordinates of a location
        """

        minx = min(self.G.nodes[node]['x'] for node in self.G.nodes)
        maxx = max(self.G.nodes[node]['x'] for node in self.G.nodes)
        miny = min(self.G.nodes[node]['y'] for node in self.G.nodes)
        maxy = max(self.G.nodes[node]['y'] for node in self.G.nodes)

        return ((minx,miny),(maxx,maxy))
    
    def plot_geography_map(self,
                           ax: Axes):
        color_road = "#d4d4d4ff"
        width_road = VehicleMarker.RADIUS_VEHICLE*2
        road_style = BoxStyle("Round", pad=0.2)
        fig, ax = ox.plot.plot_graph(
          self.G,
          ax=ax,  # optionally draw on pre-existing axis
          figsize=(8, 8),  # figure size to create if ax is None
          bgcolor="w",  # background color of the plot
          node_color="w",  # color of the nodes
          node_size=0,  # size of the nodes: if 0, skip plotting them
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

    def get_nearest_node(self, lat:float, long:float) -> Tuple[uuid.UUID,float]:
        """
        Return the nearest node to the given lat/long and the distance to it.
        """
        nearest_node,dist = ox.distance.nearest_nodes(self.G, X=long, Y=lat,return_dist=True)
        return self._loc_list[nearest_node],dist

    ### PRIVATE
    def _get_network_igraph(self) -> ig.Graph:
        G = self.G_ig.copy()
        # prepare graph to also have UUID
        G.vs['uuid'] = copy.deepcopy(self._loc_list)
        return G
    def get_distance(self,start:uuid.UUID,end:uuid.UUID):
        """
        Return the distance between two locations
        """
        idxa = self._loc_map.get(start)
        idxb = self._loc_map.get(end)

        if self.PRECOMPUTE: #Check if distances are precomputed
            return self._distance_matrix[idxa][idxb] if idxa is not None and idxb is not None else None
        else:
            path = self.G_ig.get_shortest_paths(v=idxa, to=idxb, weights=self.weight)[0]
            gdf = ox.routing.route_to_gdf(self.G, path, weight=self.weight)
            return gdf['travel_time'].sum() if len(gdf) > 0 else None # Convert to SimDuration
    def get_distance_meters(self, start:uuid.UUID, end:uuid.UUID):
        coords1 = self._locations[start].coordinate_vector
        coords2 = self._locations[end].coordinate_vector
        distance = geodesic(coords1, coords2).meters
        return distance

    def _compute_distance_matrix(self) -> np.ndarray:
        dir_path = os.path.dirname(os.path.realpath(__file__)) #get current directory
        if os.path.exists(dir_path + '/distances.npy'):
          self._distance_matrix = np.load(dir_path + '/distances.npy')  
        else:
          for i in range(0,self.n_nodes):
              print("Starting computation for node {}/{}".format(i+1,self.n_nodes))
              for j in range(0,self.n_nodes):
                  if i == j: #Skip self-loops
                      self._distance_matrix[i][j] = 0.0
                      continue
                  path = self.G_ig.get_shortest_paths(v=i, to=j, weights=self.weight)[0]
                  gdf = ox.routing.route_to_gdf(self.G, path, weight=self.weight)
                  self._distance_matrix[i][j] = gdf['travel_time'].sum() #compute distance from trajectory
        # Complete symmetry
        # self._distance_matrix += self._distance_matrix.T
        # import matplotlib.pyplot as plt
        # plt.matshow(self._distance_matrix)
        # plt.show()
      
    def _get_neighbours(self, locidx:int) -> np.ndarray:
        return np.where(self._adjacency_matrix[locidx] == 1)[0] #get all neighbours of locidx

    def get_traj(self, list_of_locations: list[uuid.UUID]) -> list[Tuple[uuid.UUID, SimDuration]]:
        """
        Converts a list of destinations into a full trjectory 
        
        Parameters
        ----------
        list_of_locations: list[uuid.UUID]
            List of locations in the order they are visited
        
        Returns
        -------
        list[Tuple[uuid.UUID, SimDuration]]
            List of tuples where each tuple contains the location and the travel time to it FROM THE PREVIOUS LOCATION
        """
        if len(list_of_locations) < 2:
            warnings.warn("List of locations is too short to compute a trajectory. Returning empty list.")
            return []
        traj = [(list_of_locations[0], SimDuration(0))] #start with first location and travel time of 0
        for i in range(1,len(list_of_locations)):
            tmp_traj = self._compute_traj(list_of_locations[i-1], list_of_locations[i]) #get trajectory from one location to the next
            traj = traj + tmp_traj[1:] #append the trajectory to the list
        return traj     

    def _compute_traj(self, start:uuid.UUID, end:uuid.UUID) -> list[tuple[uuid.UUID,SimDuration]]:
        """
        Computes the trajectory from start to end location using A* algorithm.
        """
        idxa = self._loc_map.get(start)
        idxb = self._loc_map.get(end)
        path = self.G_ig.get_shortest_paths(v=idxa, to=idxb, weights=self.weight)[0]
        if len(path) == 1:
            # If the path is just the start node, return the start node with travel time of 0
            return [(self._loc_list[idxa], SimDuration(0))]
        gdf = ox.routing.route_to_gdf(self.G, path, weight=self.weight)
        times = [SimDuration(time) for time in gdf['travel_time'].values] #convert travel times to SimDuration
        times = [SimDuration(0)] + times #set the first element to have a travel time of 0
        traj = list(zip(operator.itemgetter(*path)(self._loc_list), times)) #zip the path with the travel time
        return traj
        #A* my beloved <3
        # startInd = self._loc_map.get(start)
        # endInd = self._loc_map.get(end)
        # open_set = PriorityQueue()
        # open_set.put((0, startInd))
        # came_from: dict[int, int] = {}
        # g_score = {i: float('inf') for i in range(self.n_grid**2)}
        # g_score[startInd] = 0
        # f_score = {i: float('inf') for i in range(self.n_grid**2)}
        # f_score[startInd] = self.compute_h(start, end)
        # while not open_set.empty():
        #     current = open_set.get()[1]
        #     if current == endInd:
        #         # reconstruct path
        #         path = []
        #         while current in came_from:
        #             path.append(self._loc_list[current])
        #             current = came_from[current]
        #         path.append(self._loc_list[startInd])
        #         path.reverse()
        #     neighbours = self._get_neighbours(current)
        #     for neighbour in neighbours:
        #         tentativeGscore = g_score[current] + self.compute_h(self._loc_list[current], self._loc_list[neighbour])
        #         if tentativeGscore < g_score[neighbour]:
        #             came_from[neighbour] = current
        #             g_score[neighbour] = tentativeGscore
        #             f_score[neighbour] = tentativeGscore + self.compute_h(self._loc_list[neighbour], end)
        #             if neighbour not in [i[1] for i in open_set.queue]:
        #                 open_set.put((f_score[neighbour], neighbour))
        # #Compute Travel Time along the path
        # time = np.zeros(len(path), dtype=SimDuration)
        # for i in range(1,len(path)):
        #     # print(f"Distance from {path[i]} to {path[i+1]}: {self.get_distance(path[i],path[i+1])}")
        #     time[i] = self.predict_travel_duration(path[i-1], path[i])#This is just to compute the travel time, not used
        # time[0] = SimDuration(0)  #Set the first element to have a travel time of 0
        # time = list(time)
        # traj = list(zip(path, time)) #zip the path with the travel time
        # return traj    
    ### DEBUG ###
    def _debug_print(str: str):
        print("[ManGeo] "+str)
