from abc import ABC,abstractmethod
import uuid
from typing import ClassVar
import numpy as np
from matplotlib.axes import Axes
import copy
import warnings
import igraph as ig
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
import leidenalg as la
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import cm
# import scienceplots

from vehicle import VehicleTrajectory
from simetime import SimTime, SimDuration

class Region:
    """
    Class holding information about a region i.e. a set of nodes
    
    Attributes
    ----------
    id: uuid.UUID
        ID of the region
    locations: set[uuid.UUID]
        set of all IDs in this region
    central_location: uuid.UUID
        id of the central location
    """
    def __init__(self,
                    region_id: uuid.UUID,
                    locations: list[uuid.UUID],
                    central_location: uuid.UUID):
        self.id: uuid.UUID = region_id
        self.locations: set[uuid.UUID] = set(locations)
        self.central_location: uuid.UUID = central_location

class RegionDivision:
    """
    Class holding infromation about all the regions and how they are connected

    Attributes
    ----------
    regions: dict[uuid.UUID,Region]
        regions as a dictionary
    region_grpah: nx.Graph
        the region graph i.e. how different regions are related
        an undirected graph without selfloops.
        Regions IDs are stored on the nodes as `region_uuid`
        travel duration between central nodes is stored on the edges `travel_duration`
    location_to_region: dict
        lookup table for what location id is in what region id
    """
    def __init__(self,
                 regions: dict[uuid.UUID,Region],
                 region_graph: nx.Graph):
        self.regions: dict[uuid.UUID,Region] = regions
        self.region_graph: nx.Graph = region_graph

        self.location_to_region: dict[uuid.UUID,uuid.UUID] = self._build_loctoregion()

    def num_regions(self) -> int:
        return len(self.regions)
    
    def get_max_center_trip_time(self) -> float:
        """
        Return the maximal trip time between two centers
        """
        travel_times = [data['travel_time'] for _,_,data in self.region_graph.edges(data=True)]
        return np.max(travel_times)
    
    def get_all_region_ids(self) -> list[uuid.UUID]:
        return list(self.regions.keys())

    def plot_region_division(self, coord_getter,
                             figsize=(10, 10),
                             do_save=True):
        """
        Plots the RegionDivision.

        Parameters
        ----------
        coord_getter: callable
            A function taking (uuid) and returning np.ndarray of [x, y] coordinates.
        figsize: tuple
            Size of the figure.

        Note: written mainly by ChatGPT
        """
        # plt.style.use(['science','ieee'])
        fig, ax = plt.subplots(figsize=figsize)

        cmap = cm.get_cmap('tab20')
        region_colors = {}

        all_coords = []
        for i, (region_id, region) in enumerate(self.regions.items()):
            color = cmap(i % 20)

            # Plot all location nodes as scatter points
            coords = np.array([coord_getter(loc_id).copy() for loc_id in region.locations])
            if len(coords) > 0:
                ax.scatter(coords[:, 0], coords[:, 1],
                        color=color, s=40, label=f"Region {i}", zorder=1) #set to s=1 for good results with manhattan
                all_coords.extend(coords)

            # Highlight central location
            central_coord = coord_getter(region.central_location).copy()
            ax.scatter([central_coord[0]], [central_coord[1]],
                    color=color, edgecolor='black', linewidths=2, #set linewidth to .5 for manhattan
                    s=120, marker='o', zorder=2) #set to s=20 for good results with manhattan
            all_coords.append(central_coord)

        # Plot region graph between central nodes
        for u, v, _ in self.region_graph.edges(data=True):
            region_id_u = self.region_graph.nodes[u]['region_uuid']
            region_id_v = self.region_graph.nodes[v]['region_uuid']

            region_u = self.regions[region_id_u]
            region_v = self.regions[region_id_v]

            coord_u = coord_getter(region_u.central_location).copy()
            coord_v = coord_getter(region_v.central_location).copy()

            ax.plot([coord_u[0], coord_v[0]],
                    [coord_u[1], coord_v[1]],
                    color='gray', linewidth=.5, linestyle='--', zorder=0)

        ax.set_aspect('equal')
        ax.set_title("Region Division")
        ax.axis('off')
        if do_save:
            fig.savefig("out/region_division.png")

        plt.show(block=False)

        
    ### PRIVATE ###
    def _build_loctoregion(self) -> dict[uuid.UUID,uuid.UUID]:
        """build the lookup table location to region ID"""
        assert hasattr(self,'regions'), \
            "Define regions before trying to build a lookup table!"
        
        lookup_table = {}

        # get all locations
        locs = [] #just for sanity checking
        for region in self.regions.values():
            local_locs = list(region.locations)
            locs.extend(local_locs)
            region_id = region.id
            for loc in local_locs:
                lookup_table[loc] = region_id

        assert len(locs) == len(set(locs)), \
            "some region have the same location! they are not disjunct. This should not be the case..."
        
        return lookup_table

class Geography(ABC):
    """
    Abstract class for the geographical representation of the simulation.

    This class should be implemented by subclasses which implement the geography of the simulation.
    Each geography will have different methods to compute the distances between nodes and the storage of
    the location relationships.
    Available Geographies:
    bielegrid: Cartesian grid with n x n locations.

    Methods
    -------
    __init__():
        Initialize Geo instance.
    get_location_ids() -> set[uuid.UUID]:
        Return the location IDs as a set.
    get_location_coordinates(location_id: uuid.UUID) -> tuple:
        Returns the coordinates of a location given its ID.
    """

    DEBUG_PRINT = False

    @abstractmethod
    def __init__(self):
        """
        Initialize Geo instance.
        """
        pass
    @abstractmethod
    def get_location_ids(self)-> set[uuid.UUID]:
        """
        Return the location IDs as a set
        """
        pass
    @abstractmethod
    def get_location_coordinates(self,
                                 location: uuid.UUID,
                                 do_assert_location = True) -> np.ndarray:
        """
        Returns the coordinates for a given location

        Parameters
        ----------
        location: uuid.UUID
            desired location
        do_assert_location: bool
            whether location should be asserted, defaults to True

        Returns
        -------
        Coordinates as np array [x,y]
        """
        pass
    @abstractmethod
    def get_distance(self,start:uuid.UUID,end:uuid.UUID):
        """
        Returns the distance between two locations. This can be precomputed or computed on the fly. Depending on the geography.
        
        Note: usually use time as the distance metric
        """
        pass
    @abstractmethod
    def get_traj(self,\
                list_of_locations: list[uuid.UUID]) -> list[uuid.UUID]:
        """
        Returns a list of location IDs representing the trajectory between two locations.

        Where a trajectory are all the locations a vehicle passes.

        Parameters
        ----------
        list_of_locations: list[uuid.UUID]
                List of locations in the order they are visited

        Returns
        -------
        List of all location (including start and end) where one has to 
        pass through in order to reach end from start
        """
        pass

    @abstractmethod
    def plot_geography_map(ax: Axes):
        """
        Function to plot the background map into some axes

        Parameters
        ----------
        ax: Axes
            axes to where the background should be plotted to
        """
        raise NotImplementedError("Implement this in a subclass!")
    
    @abstractmethod
    def predict_travel_duration(self,
                                departure_location: uuid.UUID,
                                arrival_location: uuid.UUID) -> SimDuration:
        """
        Predict travel duration from one location to another location
        
        Parameters
        ----------
        departure_location: uuid.UUID
            Place where journey starts
        arrival_location: uuid.UUID
            Place where journey ends

        Returns
        -------
        SimDuration predicted duration
        """
        raise NotImplementedError("Implement this in a subclass!")
    
    @abstractmethod
    def get_distance_meters(self,
                            start: uuid.UUID,
                            end:uuid.UUID):
        """
        Return the distance between two points in meters
        Parameters
        ----------
        start: Location A
        end: Location B

        Returns
        -------
        float: Meters between two locations
        """
        raise NotImplementedError("Implement this in a subclass")
    
    def get_vehicle_coordinates_approx(self,
                                       timepoint_eval: SimTime,
                                       timepoint_anchor: SimTime,
                                       location_anchor: uuid.UUID,
                                       vehicle_traj: VehicleTrajectory,
                                       ) -> np.ndarray:
        """
        Obtain the coordinates for a given vehicle at a future times given 
        its anchor and trajectory

        Works specific geography agnostic by interpolating between the
        coordinates (assuming constant velocity)

        Computes in 2 steps:
        (1) find the next timepoint and location after evaluation and before
        (2) interpolate between the two accordin to the time difference

        Parameters
        ----------
        timepoint_eval: SimTime
            timepoint at which the coordinates should be returned = evaluation timepoint
        timpoint_anchor: SimTime
            timepoint at which the vehicle started driving at anchor
        location_anchor: uuid.UUID
            anchor location
        vehicle_traj: VehicleTrajectory
            vehicle trajectory

        Returns
        -------
        np.ndarray: [x,y] where x,y are the coordinates (in m) of the vehicle at the evaluation time
        """

        # Input Validation
        if timepoint_eval < timepoint_anchor:
            raise ValueError(f"Cannot estimate the coordinates for a time "+
                f"behind the anchor time i.e. {timepoint_eval} < {timepoint_anchor}")

        # see if single location or no time and just send that first location
        if vehicle_traj.get_num_entries() == 0\
            or timepoint_anchor == timepoint_eval:
            return copy.copy(self.get_location_coordinates(location_anchor,
                                                           do_assert_location=False))
        
        # if evaluation time is larger than the latest time
        _,time_max = vehicle_traj.get_time_range()
        if time_max < timepoint_eval:
            warnings.warn(f"Wanted to know vehicle coordinate at {timepoint_eval} "+
                          f"however the trajectory only covers up to {time_max}. "+
                          "Will return last entry. This is unwanted/unexpected behavior!")
            loc_last,_ = vehicle_traj.get_closest_point_in_time( #this is not efficient, but since this is not desired behavior and should not happen, who cares about efficiency
                time=timepoint_eval,in_future=False
            )
            return copy.copy(self.get_location_coordinates(loc_last,
                                                           do_assert_location=False))
        
        ### Step (1) ###

        # query trajectory
        loc_before, time_before = vehicle_traj.get_closest_point_in_time(
            time=timepoint_eval, in_future=False)
        loc_after, time_after = vehicle_traj.get_closest_point_in_time(
            time=timepoint_eval, in_future=True)
        
        # handle None Values
        if loc_before is None or time_before is None: #could not find something before --> am between [anchor,first point traj]
            loc_before = location_anchor
            time_before = timepoint_anchor
        assert loc_after is not None and time_after is not None, \
            "How come you find no location after even though the trajectory is not empty "+\
            "and your evaluation time is smaller than the maximal time?"

        ### Step (2) ###

        #determine linear interpolation
        t_step = time_after.time_s - time_before.time_s
        t_tillevaluation = timepoint_eval.time_s - time_before.time_s
        scale = t_tillevaluation/t_step

        #do interpolation
        coord_before = self.get_location_coordinates(loc_before,
                                                     do_assert_location=False)
        coord_after = self.get_location_coordinates(loc_after,
                                                    do_assert_location=False)
        return (coord_after-coord_before)*scale + coord_before
    
    @abstractmethod
    def _get_network_igraph(self) -> ig.Graph:
        """
        Obtain the network as an graph in the igraph package

        Return
        ------
        None if this is not possible yet, otherwise the igraph of the network

        Vertices of igraph with metadata:
            `uuid` ... lcoation UUID
            `x` ... x coordinate
            `y` ....y coordinate
        Edges of igraph with metadata
            `travel_time` ... travel time in s
        """
        raise NotImplementedError("Implement this function in subclass!")
    
    def get_regions(self,
                    k_min: int) -> RegionDivision:
        """
        Computes Regions for the geography.

        Parameters
        ----------
        k_min: int
            minimal amount of regions

        Returns precomputed values if available, otherwise does live computation
        """
        if Geography.DEBUG_PRINT:
            Geography._debug_print("Computing Regions...")

        G = self._get_network_igraph()
        assert G.is_connected(mode="STRONG"), "Network Graph is not strongly connected, funny map you have there..."

        # labels_spectral = self._spectral_clustering(G,k=10,normalized=True) #tune k
        # could also fnd regions differntly

        regions_leiden = self._leiden(G,
                                      k_min=k_min)

        return regions_leiden
    
    def _spectral_clustering(self,
                             g, k=2, normalized=True):
        """
        Perform spectral clustering on a directed igraph.Graph object.

        DISCLAIMER: NOT WORKING, MORE PROOF OF CONCEPT via ChatGPT
        
        Parameters
        ----------
        g (igraph.Graph): A directed graph
        k (int): Number of clusters
        normalized (bool): Whether to use the normalized Laplacian
        
        Returns
        -------
        labels (np.ndarray): Cluster labels for each node
        """
        # Step 1: Get adjacency matrix (dense or sparse)
        A = np.array(g.get_adjacency().data, dtype=float)

        # Step 2: Symmetrize the adjacency matrix
        A_sym = (A + A.T) / 2

        # Step 3: Degree matrix
        degrees = A_sym.sum(axis=1)
        D = np.diag(degrees)

        # Step 4: Compute Laplacian
        if normalized:
            # Avoid divide by zero for isolated nodes
            with np.errstate(divide='ignore'):
                D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # handle division by zero
            L = np.eye(len(A)) - D_inv_sqrt @ A_sym @ D_inv_sqrt
        else:
            L = D - A_sym

        # Step 5: Compute the first k eigenvectors
        # For small graphs, use dense eigendecomposition
        eigvals, eigvecs = eigsh(L, k=k, which='SM')  # 'SM' = smallest magnitude

        # Step 6: Normalize rows (only for normalized Laplacian)
        U = eigvecs
        if normalized:
            row_norms = np.linalg.norm(U, axis=1, keepdims=True)
            U = np.divide(U, row_norms, where=row_norms != 0)

        # Step 7: Apply k-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(U)

        return labels

    def _leiden(self,
                graph: ig.Graph,
                k_min: int = 5) -> RegionDivision:
        """"
        Find Regions according to the Leiden Algorithm

        Parameters
        ----------
        graph: ig.Graph
            Graph to work on
            Must have an edge field `travel_time`
        k_min: int = 5
            Minimal amount of regions
        """
        if Geography.DEBUG_PRINT:
            Geography._debug_print("Finding Regions according to the Leiden Algorithm")

        assert k_min < graph.vcount(), \
            f"Duuuude, you want at least as many regions ({k_min}) as there is locations ({graph.vcount()}). Check your logic."

        # write out similarities by inversing the weights
        graph.es['similarity'] = [1.0 / w if w != 0 else 0 for w in graph.es['travel_time']]

        ### find partitions ###
        found_partion = False
        resolution_parameter = 0.001 #tune here for a better init value
        iteration_scale = 1 #do not change
        iteration_scale_delta = 0.01 #tune for faster convergence, the more agressive the smaller
        while(not found_partion):
            partition = la.find_partition(graph,
                                        la.CPMVertexPartition,
                                        weights='similarity',
                                        resolution_parameter=resolution_parameter) # tune resolution parameter!
            iteration_scale += iteration_scale_delta
            # check size: if too small
            if len(partition) < k_min:
                resolution_parameter *= (1+0.1/iteration_scale)
                if Geography.DEBUG_PRINT:
                    Geography._debug_print(f"Found a partition with to few regions ({len(partition)})."+\
                                           f" Trying again with a resolution parameter of {resolution_parameter}.")
                continue
            #check size: if just single vertex cover
            if len(partition) == graph.vcount():
                resolution_parameter *= (1-0.06/iteration_scale)
                if Geography.DEBUG_PRINT:
                    Geography._debug_print(f"Found a partition that just gives every location a region. Retrying with resolution parameter {resolution_parameter}.")
                continue
            
            #check connected --> regions do not need to be strongly connected!
            found_partion = True
            # for subgraph in partition.subgraphs():
            #     if not subgraph.is_connected(mode="STRONG"):
            #         found_partion = False
            #         resolution_parameter *= (1-0.08/iteration_scale)
            #         if Geography.DEBUG_PRINT:
            #             Geography._debug_print(f"Found partition has subgraphs that are not strongly connected! Retry with a resolution aprameter of {resolution_parameter}")
            #         break
        
        if Geography.DEBUG_PRINT:
            Geography._debug_print(f"Found a partition with {len(partition)} regions after {int((iteration_scale-1)/iteration_scale_delta)} iterations.")

        ### cast partitions into regions ###

        # generate regions
        regions = {}
        for subgraph in partition.subgraphs():
            # extract location IDs
            location_ids = [v['uuid'] for v in subgraph.vs]
            # decide on a central node based on 
            centralities = subgraph.closeness()
            central_node = subgraph.vs[centralities.index(max(centralities))]
            central_location_id = central_node['uuid']
            # build object
            region_id = uuid.uuid4()
            regions[region_id] = Region(
                region_id=region_id,
                locations=location_ids,
                central_location=central_location_id
            )

        # generate region graph
        region_graph: ig.Graph = graph.copy()
        region_graph.contract_vertices(partition.membership, combine_attrs=dict(
            weight="sum",
            size="mean",
        ))
        region_graph.simplify(combine_edges="sum", loops=True) #remove self loops, sum up weights
        region_graph = region_graph.as_undirected(combine_edges="mean") #just take the mean if edges are combined
        # region_graph['region_uuid'] = [key for key in regions] #write out what region this is into the graph --> is not transformed as it is an object
        
        # prepare network_x graph
        nx_region_graph = region_graph.to_networkx()
        region_uuids = [key for key in regions]
        for node_id, region_id in zip(nx_region_graph.nodes, region_uuids):
            nx_region_graph.nodes[node_id]['region_uuid'] = region_id
        for u, v, data in nx_region_graph.edges(data=True):
            uuid_u = nx_region_graph.nodes[u].get("region_uuid")
            uuid_v = nx_region_graph.nodes[v].get("region_uuid")
            travel_duration = self.predict_travel_duration(departure_location=regions[uuid_u].central_location,
                                                           arrival_location=regions[uuid_v].central_location)
            data["travel_duration"] = travel_duration
        # write actual region division object
        region_division = RegionDivision(
            regions=regions,
            region_graph=nx_region_graph
        )

        return region_division

    ### DEBUG ###
    def _debug_print(str: str):
        print("[GEO] "+str)