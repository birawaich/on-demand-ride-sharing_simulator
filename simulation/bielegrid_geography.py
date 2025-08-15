import uuid
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import warnings
import copy
from queue import PriorityQueue
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, BoxStyle
import igraph as ig

from visualization import VehicleMarker
from geo import Geography
from simetime import SimDuration


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
    

class BieleGridGeography(Geography):
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
    def __init__(self, n_grid: int, precompute_dist: bool=True):
        assert n_grid >= 1, "grid size must be at least 1"
        self.n_grid = n_grid

        # Map of location ids to index in the adjacency matrix and distance matrix
        self._loc_list: list[uuid.UUID] = []
        self._loc_map: dict[uuid.UUID,int] = {} #Map of index to location id
        # Adjacency matrix 
        self._adjacency_matrix: np.ndarray = np.zeros((n_grid**2,n_grid**2), dtype=np.int8)
        # Distance matrix
        self._distance_matrix: np.ndarray = np.zeros((n_grid**2, n_grid**2), dtype=np.float16)
        # set vehicle speed
        self._vehicle_speed_mps: float = 50/3.6

        self._grid_seperation: float = 1e3

        locations: dict[uuid.UUID,PointInformation] = {}
        # assign uuid to coordinates
        for x in range (1,n_grid+1): #make it 1 based
            for y in range(1,n_grid+1):
                local_uuid = uuid.uuid4()
                self._loc_list.append(local_uuid) #add to map
                self._loc_map.update({local_uuid: len(self._loc_list)-1}) #add to map
                locations.update({local_uuid: #uuid as key
                                 PointInformation( #point information as value
                                     id=local_uuid,
                                     coordinate_x=x,
                                     coordinate_y=y,
                                     coordinate_vector=np.array([x*self._grid_seperation
                                                                 ,y*self._grid_seperation]
                                                                ,dtype=float),
                                     name=f"({x},{y}) {local_uuid}")
                                })
        #Deprecated: precompute distances for each location to all other locations        
        # for loc in locations.keys():
        #     locations[loc].dist_to_nodes = self._compute_distances_dict(loc,locations) if precompute_dist else {} 
        self._locations: dict[uuid.UUID,PointInformation] = locations #location lookup table
        # compute distance matrix
        self._init_adjacency_matrix() #initialize adjacency matrix
        self._compute_distance_matrix()
        # print(f"Distance matrix computed as {self._distance_matrix}")
        # input("Press Enter to continue...") #debugging, remove later
        
        # test_traj = self.get_traj([self._loc_list[0],self._loc_list[n_grid],self._loc_list[-1]]) #debugging, remove later
        # # test_traj = self.get_traj(self._loc_list[0],self._loc_list[-1]) #debugging, remove later
        # print(f"Test trajectory from {self._loc_list[0]} to {self._loc_list[-1]}: {test_traj}")
        # input("Press Enter to continue...") #debugging, remove later

    ### PUBLIC

    def get_location_ids(self) -> set[uuid.UUID]:
        """
        Return the location IDs as a set
        """
        return set(self._locations.keys())
    
    def predict_travel_duration(self,
                                departure_location: uuid.UUID,
                                arrival_location: uuid.UUID) -> SimDuration:
        # distance = np.linalg.norm(self._locations[arrival_location].coordinate_vector
        #                           - self._locations[departure_location].coordinate_vector,
        #                           ord=1)
        distance = self._grid_seperation * self.get_distance(departure_location,arrival_location) #in m
        duration = SimDuration(distance/self._vehicle_speed_mps)

        # if BieleGridGeography.DEBUG_PRINT: #disabled as this is used a lot
        #     BieleGridGeography._debug_print(
        #         f"Calculated duration from {self._locations[departure_location].name} "
        #           +f"to {self._locations[arrival_location].name} to be {duration}.")

        return duration
    
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

        # demo code for non grids
        # coord_min = np.full(2,np.inf)
        # coord_max = np.full(2,np.inf)

        # for location in self._locations.values():
        #     for idx_dim in [0,1]:
        #         if coord_min[idx_dim] > location.coordinate_vector[idx_dim]:
        #             coord_min[idx_dim]=location.coordinate_vector[idx_dim]
        #         elif coord_max[idx_dim] < location.coordinate_vector[idx_dim]:
        #             coord_max[idx_dim] = location.coordinate_vector[idx_dim]

        coord_min = np.full(2,1*self._grid_seperation)
        coord_max = np.full(2,self.n_grid*self._grid_seperation)

        return (coord_min,coord_max)
    
    def plot_geography_map(self,
                           ax: Axes):
        color_road = "#d4d4d4ff"
        width_road = VehicleMarker.RADIUS_VEHICLE*2
        road_style = BoxStyle("Round", pad=0.2)

        for x in range(1,self.n_grid+1):
            #vertical
            street_vert = FancyBboxPatch(
                    (x*self._grid_seperation - width_road*0.5,
                        self._grid_seperation - width_road*0.5),
                    width_road,self._grid_seperation*(self.n_grid-1)+width_road,
                    facecolor=color_road,edgecolor='none', linewidth=0,
                    boxstyle=road_style,mutation_scale=width_road*2)
            ax.add_patch(street_vert)
            #horizontal
            street_hor = FancyBboxPatch(
                    (self._grid_seperation - width_road*0.5,
                        x*self._grid_seperation - width_road*0.5),
                    self._grid_seperation*(self.n_grid-1)+width_road,width_road,
                    facecolor=color_road,edgecolor='none',  linewidth=0,
                    boxstyle=road_style,mutation_scale=width_road*2)
            ax.add_patch(street_hor)
            

    ### PRIVATE

    def _get_network_igraph(self) -> ig.Graph:

        # build graph
        time_to_all_nodes = self._distance_matrix*self._grid_seperation/self._vehicle_speed_mps
        G = ig.Graph.Weighted_Adjacency(matrix=time_to_all_nodes*self._adjacency_matrix,
                                        mode='directed')
        # https://igraph.org/python/versions/latest/api/igraph.GraphBase.html#_Weighted_Adjacency

        # set meta data
        G.vs['uuid'] = self._loc_list
        G.vs['x'] = [self._locations[loc].coordinate_vector[0] for loc in self._loc_list]
        G.vs['y'] = [self._locations[loc].coordinate_vector[1] for loc in self._loc_list]
        G.es['travel_time'] = [w for w in G.es['weight']]

        # test: graphml only likes basic types...
        # G.vs['x'] = [float(self._locations[loc].coordinate_vector[0]) for loc in self._loc_list]
        # G.vs['y'] = [float(self._locations[loc].coordinate_vector[1]) for loc in self._loc_list]
        # G.write_graphml("out/tmp.graphml")
        return G

    def _init_adjacency_matrix(self):
        """
        Initialize the adjacency matrix for the BieleGridGeography.
        The adjacency matrix is a square matrix of size n_grid x n_grid,
        where n_grid is the number of locations in the grid.
        """
        a = np.diag(np.ones(self.n_grid - 1,dtype=np.int8), k=1) + np.diag(np.ones(self.n_grid - 1,dtype=np.int8), k=-1)
        I = np.eye(self.n_grid)
        A_rows = np.kron(a, I)
        # Connections along the 'columns' of the grid
        A_cols = np.kron(I, a)

        # The total adjacency matrix is the sum of connections in both directions
        self._adjacency_matrix = A_rows + A_cols

    def get_distance(self,start:uuid.UUID,end:uuid.UUID):
        """
        Return the distance between two locations
        """
        idxa = self._loc_map.get(start)
        idxb = self._loc_map.get(end)
        # print(f"Getting distance from {start} to {end} with indices {idxa} and {idxb}")
        # input("Waiting for user input to continue...") #debugging
        return self._distance_matrix[idxa][idxb] if idxa is not None and idxb is not None else None

        if bool(self._locations[start].dist_to_nodes): #Check if dictionary is empty(i.e. distances not precomputed)
            return self._locations[start].dist_to_nodes[end]
        else:
            #TODO Implement A* to compute distance to node
            return 2
    
    def get_distance_meters(self, start:uuid.UUID, end:uuid.UUID):
        return self.get_distance(start,end)
    
    def _compute_distance_matrix(self) -> np.ndarray:
        self._distance_matrix = np.zeros((self.n_grid**2, self.n_grid**2), dtype=np.float16)
        for i in range(0,self.n_grid**2):
            p1 = self._loc_list[i]
            for j in range(i+1,self.n_grid**2):
                p2 = self._loc_list[j]
                traj = self.get_traj([p1,p2]) #get trajectory from p1 to p2
                self._distance_matrix[i][j] = self._compute_distance_from_trajectory(traj) #compute distance from trajectory
                # v2 = locations[j].coordinate_vector
                # self._distance_matrix[i][j] = np.linalg.norm(v1-v2)
        # Complete symmetry
        self._distance_matrix += self._distance_matrix.T
        # import matplotlib.pyplot as plt
        # plt.matshow(self._distance_matrix)
        # plt.show()

    def _compute_distance_from_trajectory(self, trajectory: list[(uuid.UUID,SimDuration)]) -> float:
        """
        Compute the distance of a trajectory given as list of locations
        """
        if len(trajectory) < 2:
            warnings.warn("Trajectory is too short to compute distance. Returning 0.")
            return 0.0
        dist = 0.0
        idx1 = self._loc_map.get(trajectory[0][0])
        for i in range(1, len(trajectory)):
            idx2 = self._loc_map.get(trajectory[i][0])
            dist += self._adjacency_matrix[idx1][idx2]
            idx1 = idx2
        return dist
    
    def compute_h(self, start:uuid.UUID, end:uuid.UUID,heuristic=1) -> int:
        """
        Can manually switch between heuristics
        Heuristics:
            0 -> L_1
            1 -> L_2
            2 -> L_inf
        """
        v1 = self._locations[start].coordinate_vector
        v2 = self._locations[end].coordinate_vector
        if heuristic == 0:
            dist = np.linalg.norm(v1-v2, ord=1) #L_1 norm
        elif heuristic == 1:
            dist = np.linalg.norm(v1-v2, ord=2) #L_2 norm
        elif heuristic == 2:
            dist = np.linalg.norm(v1-v2, ord=np.inf) #L_inf norm
        return dist
      
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
        startInd = self._loc_map.get(start)
        endInd = self._loc_map.get(end)
        open_set = PriorityQueue()
        open_set.put((0, startInd))
        came_from: dict[int, int] = {}
        g_score = {i: float('inf') for i in range(self.n_grid**2)}
        g_score[startInd] = 0
        f_score = {i: float('inf') for i in range(self.n_grid**2)}
        f_score[startInd] = self.compute_h(start, end)
        while not open_set.empty():
            current = open_set.get()[1]
            if current == endInd:
                # reconstruct path
                path = []
                while current in came_from:
                    path.append(self._loc_list[current])
                    current = came_from[current]
                path.append(self._loc_list[startInd])
                path.reverse()
            neighbours = self._get_neighbours(current)
            for neighbour in neighbours:
                tentativeGscore = g_score[current] + self.compute_h(self._loc_list[current], self._loc_list[neighbour])
                if tentativeGscore < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentativeGscore
                    f_score[neighbour] = tentativeGscore + self.compute_h(self._loc_list[neighbour], end)
                    if neighbour not in [i[1] for i in open_set.queue]:
                        open_set.put((f_score[neighbour], neighbour))
        #Compute Travel Time along the path
        time = np.zeros(len(path), dtype=SimDuration)
        for i in range(1,len(path)):
            # print(f"Distance from {path[i]} to {path[i+1]}: {self.get_distance(path[i],path[i+1])}")
            time[i] = self.predict_travel_duration(path[i-1], path[i])#This is just to compute the travel time, not used
        time[0] = SimDuration(0)  #Set the first element to have a travel time of 0
        time = list(time)
        traj = list(zip(path, time)) #zip the path with the travel time
        return traj    
    ### DEBUG ###
    def _debug_print(str: str):
        print("[BieleGeo] "+str)
