from typing import Any
import uuid
import numpy as np
import copy
import networkx as nx
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from controller import Controller
from event import Event, ControllerTimingTick
from action import Action
from environment import Environment
from vehicle import VehicleState, VehiclePlan, VehiclePlanEntry
from metrics import MetricPlot

class CombinedController(Controller):
    """
    Controller that has a main controller as well as a rebalancing component.

    Attributes
    ----------
    _main_controller: Controller
        main controller to handle everything besides rebalancing
    _rebalancing_controller: RebalancingController
        rebalancing controller
    """

    STORE_REBALANCINGMETRICS = False
    """Store the metrics collected by the rebalancing controller. If stored, also creates a plot."""

    def __init__(self,
                 main_controller: Controller,
                 env: Environment,
                 num_regions_min: int = 5):
        """
        Parameters
        ----------
        main_controller: Controller
            controller to handle all events that are not rebalancing
        env: Environment
            reference to environment
        num_regions_min: int
            minimal amount of regions
        """
        # set up main controller
        self._main_controller = main_controller
        # set up rebalancing controller
        self._rebalancing_controller = RebalancingController(env=env,
                                                             num_regions_min=num_regions_min)
        
    def process_event(self,
                      event: Event) -> Action:
        """
        Processes an avent and generates an action  
        = Control Loop
        """
        action = None
        if event is None:
            return action
        
        if isinstance(event, ControllerTimingTick)\
            and event.identification == self._rebalancing_controller.get_timing_event_identification():
            action = self._rebalancing_controller.process_event(event)
            #Sync vehicle Plans
            # return Action()
            if action is not None:
                self._main_controller.update_vehicle_plans(action.vehicle_plans)
            return action

        action = self._main_controller.process_event(event=event)

        return action
    def update_vehicle_plans(self,
                             vehicle_plans: dict[uuid.UUID, VehiclePlan]):
        """
        Update the vehicle plan if more than one controller is acting and this controller 
        needs to keep track of vehicle plans
        """
        pass
    def finalize(self):
        """finalization hook, to be called _after_ simulation"""
        if CombinedController.STORE_REBALANCINGMETRICS:
            self._rebalancing_controller.save_storage(do_plot=True)

class RebalancingController(Controller):
    """
    Rebalancing controller: just does rebalancing!

    Needs timing events with its ID.

    Do NOT instanciate on its own (unless for testing), use 
    a `CombinedController` to give any controller this rebalancing feature

    Attributes
    ----------
    _timing_event_identification: uuid.UUID
        (private) identification for the timing events
    _region_division: RegionDivision
        (private) how the regions are devided
    _env: Environment
        (private) Reference to the environment
    _T_trip: float
        (private) expected time it takes for a trip from one region to the other
    _vehicle_ids: list[uuid.UUID]
        (private) vehicle IDs
    _rng: 
        (private) random number generator
    _storage_timestamps
        (private) store the timestamps in s
    _storage_spots
        (private) store the amount of spots in each region (np.ndarray)
    _storage_empty_vehicles
        (private) store the amount of free vehicles in each reagion (np.ndarray)
    _storage_nodeid_to_regionid
        (private) mapping of the position in the array to the region id
    """

    DEBUG_PRINT = False

    PLOT_REGIONS = False
    """Set to True to plot the regions"""

    SAVE_REGION_PLOT = False
    """Set to True to store a plot of the regions used by the rebalancing controller"""

    DO_REBALANCING = True
    """Enable or disable rebalancing. Useful to record spead over regions without doing rebalancing."""

    def __init__(self,
                 env: Environment,
                 num_regions_min: int = 5):
        """
        Parameters
        ----------
        env: Environment
            reference to environment (to know about vehicles and to talk to geography)
        num_regions_min: int
            minimal amount of regions
        """
        self._env = env
        self._timing_event_identification = uuid.uuid4()

        # get partitions --> do partitions in geography!
        geo = env.get_geography()
        self._region_division = geo.get_regions(k_min=num_regions_min)
        # debug plot
        if RebalancingController.PLOT_REGIONS:
            self._region_division.plot_region_division(
                coord_getter=geo.get_location_coordinates,
                do_save=RebalancingController.SAVE_REGION_PLOT
            )

        # set up timing itervals
        self._T_trip = self._region_division.get_max_center_trip_time()
        env.insert_timing_events(time_period=self._T_trip,
                                 identification=self._timing_event_identification)
        
        # get further thigns
        self._vehicle_ids = list(env.get_vehicles_ids())
        self._rng = np.random.default_rng(seed=42)

        # set up storage
        self._storage_timestamps = []
        self._storage_spots = []
        self._storage_empty_vehicles = []
        self._storage_nodeid_to_regionid = []

    def get_timing_event_identification(self) -> uuid.UUID:
        return self._timing_event_identification
    
    def process_event(self,
                      event: Event) -> Action:
        """
        Processes an avent and generates an action  
        = Control Loop
        """

        assert isinstance(event,ControllerTimingTick), \
            "Rebalacing controller got something other than a timing tick, this is wrong!"
        if RebalancingController.DEBUG_PRINT:
            RebalancingController._debug_print(f"Got Timing Tick at {event.timepoint}")

        ### count amount of free seats and idling & free vehicles ###
        N_regions = self._region_division.num_regions()
        spots_val = np.zeros(N_regions,dtype=np.int_) #amount of spots per region (free and flexible)
        vehicles_vals = np.zeros(N_regions,dtype=np.int_) #amount of free and idling vehicles per region

        # deterime timepoint
        eval_timepoint = copy.copy(event.timepoint)
        # eval_timepoint.add_duration(SimDuration(self._T_trip)) #evaluate in the future! #TODO to implement this, need to also know amount of passengers in the future!
        
        # find relevant data of all relevant vehicle
        regionid_to_nodeindex = {data['region_uuid']: node for node,data in self._region_division.region_graph.nodes(data=True)} #Note: HAS TO CORRESPOND WITH THE INDEX OF THE NODES IN THE GRAPH
        nodeindex_to_regionid = {v:k for k,v in regionid_to_nodeindex.items()} #inverse map
        free_idling_vehicle_in_region = {key: [] for key in regionid_to_nodeindex} #what vehile (by id) is in this region
        vehicle_capacities = []
        for vehicle_id in self._vehicle_ids:
            vehicle = self._env.get_vehicle(vehicle_id)
            vehicle_capacities.append(vehicle.max_passengers)
            if vehicle.max_passengers == len(vehicle.passengers): #vehicle is not of interest if full
                continue
            # find location (or next location) of vehicle anbd corresponding region
            if vehicle.state == VehicleState.IDLING:
                location_id = vehicle.anchor_location
            else:
                location_id,_ = vehicle.traj.get_closest_point_in_time(time=eval_timepoint,in_future=bool)
            region_id = self._region_division.location_to_region[location_id]
            region_index = regionid_to_nodeindex[region_id]
            # write data
            spots_val[region_index] += vehicle.max_passengers - len(vehicle.passengers)
            if vehicle.state == VehicleState.IDLING:
                vehicles_vals[region_index] += 1
                free_idling_vehicle_in_region[region_id].append(vehicle)

        assert np.all(np.array(vehicle_capacities) == vehicle_capacities[0]), \
            "The update functions assume vehicles with the same capacities. Can implement differently, but need to think a bit (brute force problem gets a bit bigger and need more information)."
        vehcile_capacity = vehicle_capacities[0]

        # store values
        self._storage_timestamps.append(event.timepoint.time_s)
        self._storage_spots.append(copy.copy(spots_val))
        self._storage_empty_vehicles.append(copy.copy(vehicles_vals))
        self._storage_nodeid_to_regionid.append(nodeindex_to_regionid)

        if not RebalancingController.DO_REBALANCING:
            return None

        ### MAIN LOOP ###
        # (1) Select Edges
        edges = self._select_edges_delta_seats(G=self._region_division.region_graph,
                                               vehicles_empty=vehicles_vals,
                                               spots=spots_val,
                                               norm="L2")
        # (2) Update Edges
        scheduling_requests = []
        for edge in edges:
            request = self._update_edge_spots(edge=edge,
                                              spots=spots_val,vehicles_empty=vehicles_vals,
                                              vehicle_capacity=vehcile_capacity)
            if request is not None:
                scheduling_requests.append(request)

        if RebalancingController.DEBUG_PRINT:
            RebalancingController._debug_print(f"Have {len(scheduling_requests)} Scheduling Requests!")
            for request in scheduling_requests:
                RebalancingController._debug_print(f"\tMove {request['amount']} from region {request['src']} --> {request['dst']}")

        ### translate results back to an action ###
        vehicle_plans = dict()
        vehicle_states = dict()
        for request in scheduling_requests:
            if request['amount'] == 0:
                continue
            amount = request['amount']
            src_id = nodeindex_to_regionid[request['src']]
            dst_id = nodeindex_to_regionid[request['dst']]
            src_loc = self._region_division.regions[src_id].central_location
            dst_loc = self._region_division.regions[dst_id].central_location
            assert len(free_idling_vehicle_in_region[src_id]) >= amount, \
                f"Do want to move {amount} empty and idling vehicles from region {src_id} but only have {len(free_idling_vehicle_in_region[src_id])} vehicles there!"
            for i in range(amount):
                vehicle_id = free_idling_vehicle_in_region[src_id][i].id
                vehicle_states[vehicle_id] = VehicleState.DRIVING
                vehicle_plans[vehicle_id] = VehiclePlan([
                    # VehiclePlanEntry(src_loc,set()),
                    VehiclePlanEntry(dst_loc,set())
                ])
        
        return Action(vehicle_plans=vehicle_plans,
                      vehicle_states=vehicle_states)
    
    def save_storage(self,
                     do_plot = False):
        """
        Save the collected storage, do plot if set to true
        """
        # assert that lists have same length
        l = len(self._storage_timestamps)
        assert l == len(self._storage_spots), "Wrong amount!"
        assert l == len(self._storage_empty_vehicles), "Wrong amount!"
        assert l == len(self._storage_nodeid_to_regionid), "Wrong amount!"

        # prepare dictionary
        storage = {
            "timestamps": self._storage_timestamps,
            "spots": self._storage_spots,
            "empty_vehicles": self._storage_empty_vehicles,
            "nodeid_to_regionid": self._storage_nodeid_to_regionid
        }

        # store
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"out/rebalancing_controller_{timestamp}.pkl", "wb") as f:
            pickle.dump(storage, f)

        if do_plot:
            self._plot_storage()
        

    ### PRIVATE ###

    def _plot_storage(self):
        """plot a figure for the storage"""
        # set up figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.canvas.manager.set_window_title("Rebalancing Controller")

        # get data
        num_regions = self._region_division.num_regions()
        num_datapoints = len(self._storage_timestamps)
        data_spots = np.empty((num_regions,num_datapoints))
        data_timestamps = np.array(self._storage_timestamps)
        for i, spots in enumerate(self._storage_spots):
            data_spots[:,i] = spots

        l1_errors_spots = np.abs(data_spots - np.mean(data_spots,axis=0))

        # plot
        self._plot_2d_data_spread(
            ax=ax, timestamps=data_timestamps, data=l1_errors_spots,
            color='blue', desc=f"L1 Error"
        )

        # cosmetics
        MetricPlot._set_axis_for_simtime(ax,which='x')
        ax.set_ylabel('L1 Error')
        ax.set_title("Evolution of Free Spot Spread over Regions")
        ax.legend()
        ax.grid(True)

        fig.tight_layout() #fix overlaps

        # show and store
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.savefig(f"out/rebalancing_controller_{timestamp}.png", dpi=300)

        plt.show(block=False)
        plt.pause(0.01)
        

    def _plot_2d_data_spread(self,
                             ax: Axes, timestamps, data, color: str,desc: str):
        # prepare data
        min_err = np.min(data, axis=0)         # shape: (n_timepoints,)
        max_err = np.max(data, axis=0)         # shape: (n_timepoints,)
        qlow = np.quantile(data, 0.05, axis=0)  # shape: (n_timepoints,)
        qhigh = np.quantile(data, 0.95, axis=0)
        mean = np.mean(data,axis=0)

        # plot
        ax.fill_between(timestamps, qlow, qhigh, alpha=0.1, label='5-95% Quantile', color=color)
        ax.plot(timestamps, mean, label=f"Mean {desc}", color=color)
        ax.plot(timestamps, min_err, linestyle=':', color=color, alpha=0.6, label='Min/Max')
        ax.plot(timestamps, max_err, linestyle=':', color=color, alpha=0.6)

        

    def _select_edges_uniform(self,
                              G: nx.Graph,
                              vehicles_empty: np.ndarray) -> list[tuple[int,int]]:
        """
        Select edges in a round uniformly

        Parameters
        ----------
        G: nx.Graph
            the graph to operate on
        vehicles_empty: np.ndarray
            index that says in what node there is an empty vehicle        
        """
        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # assume uniform probability that the edge is selected; greedily select an edge, then remove all connecting edges of the vertex
        while(len(edges_with_vehicles)>0):
            edge = self._rng.choice(a=edges_with_vehicles,size=1,replace=False)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            edges_with_vehicles = [
                (i, j) for i, j in edges_with_vehicles
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]

        return selected_edges
    
    def _select_edges_delta_seats(self,
                                  G: nx.Graph,
                                  vehicles_empty: np.ndarray,
                                  spots: np.ndarray,
                                  norm: str = "L2") -> list[tuple[int,int]]:
        """
        Select edges where edges with a higher spot delta are more likely
        
        Uses the selected norm (either L1 or L2) as the measure. Gives non-zero probabilities to all edges!
        
        Parameters
        ----------
        G: nx.Graph
            the graph to operate on
        vehicles_empty: np.ndarray
            index that says in what node there is an empty vehicle
        spots: np.ndarray
            amount of free spots in each node
        norm: str
            either "L2" or "L1" to calculate the probabilities
        """

        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # build probabilities
        probabilities = np.zeros(shape=len(edges_with_vehicles))
        for i, edge in enumerate(edges_with_vehicles):
            if norm == "L2":
                probabilities[i] = max((spots[edge[0]] - spots[edge[1]])**2,0.5) #ensure non-zero probability by counting a spot difference of 0 as 1/2 
            elif norm == "L1":
                probabilities[i] = max(np.abs(spots[edge[0]] - spots[edge[1]]),0.5)
            else:
                raise NotImplementedError(f"Unknown Norm {norm}")
        # draw until cannot draw anymore
        while(len(edges_with_vehicles)>0):
            probabilities = probabilities / np.sum(probabilities, dtype=float) #normalize, need to do at every step as elements are removed
            edge = self._rng.choice(a=edges_with_vehicles,size=1,replace=False,p=probabilities)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            index_allowed_edges = [
                index for index, (i,j) in enumerate(edges_with_vehicles)
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]
            probabilities = probabilities[index_allowed_edges]
            edges_with_vehicles = [edges_with_vehicles[i] for i in index_allowed_edges]
        return selected_edges

    def _update_edge_spots(self,
                           edge: tuple[int,int],
                           spots: np.ndarray,
                           vehicles_empty: np.ndarray,
                           vehicle_capacity: int) -> None | dict[str, Any]:
        """
        Update a single edge and return the deltas in vehicles

        Parameters
        ----------
        edge: tuple[int,int]
            what edge, contains indexes of nodes (=regions)
        spots: np.ndarray
            amount of free spots at each node =region
        vehicles_empty
            amount of empty and idling vehicles at each node =region
        vehicle_capacity
            capacity of a vehicle

        Returns
        -------
        dictionary that describes an order or None if there is no order
            'src' ... source index
            'dst' ... destination index
            'amount' ... amount of vehicles to move
        """
        i = edge[0]
        j = edge[1]

        delta_s = np.abs(spots[i] - spots[j])

        # do action
        if delta_s == 0: #equal values # Ensures (P1)
            # print(f"Edge ({i},{j}) is already balanced in terms of seats!")
            return None

        if spots[i] > spots[j]: #find direction
            node_src = i
            node_dst = j
        else:
            node_src = j
            node_dst = i
        delta_v = vehicles_empty[node_src] - vehicles_empty[node_dst]
       
        # just move to get the smallest delta, enforces (P3)
        # implemented as brute force
        move_best = 0
        move_best_value = delta_s
        for move in range(1,delta_v+1):
            move_value = np.abs(spots[node_src]-move*vehicle_capacity - (spots[node_dst]+move*vehicle_capacity))
            if move_value <= move_best_value:
                move_best_value = move_value
                move_best = move
        # move by best move: generate return dict
        return {
            'src': node_src,
            'dst': node_dst,
            'amount': move_best
        }

    ### DEBUG ###
    def _debug_print(str: str):
        print("[RebalCtr] "+str)
