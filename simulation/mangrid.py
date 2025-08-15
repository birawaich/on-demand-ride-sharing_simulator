import numpy as np
import uuid
import pandas as pd
import datetime as dt
import pickle as pkl

from environment import Environment
from event import Event, RideRequest
from action import Action
from simetime import SimTime, SimDuration
from vehicle import Vehicle, VehicleState
from passenger import Passenger
from metrics import MetricMaster
from matplotlib.pyplot import gca
from man_geography import ManGridGeography
from visualization import EnvironmentVisualization

class ManGrid(Environment):
    """
    Environment Class representing ManGrid (Manhattan NY,NY, USA) as a grid-like environment.
    """

    # CLASS PROPERTIES
    DEBUG_PRINT = False

    def __init__(self,
                 sim_duration_s=1):
        

        ### DEFAULTS (pass in as arguments if this should be altered)
        num_vehicles = 5 #amount of vehicles
        vehicle_capacity = 4 #amount of seats in a vehicle
        ### END DEFAULTS
        if ManGrid.DEBUG_PRINT:
            ManGrid._debug_print("Setting up ManGrid Environment...")
        # set up geography
        self._geo = ManGridGeography(precompute_dist=True)

        super().__init__(sim_duration_s)
        
        # set up random generator
        self._rng = np.random.default_rng(0)

        # set object properties (pass to constructor if needed)      

        self._setup_vehicles(num_vehicles=num_vehicles,
                             vehicle_capacity=vehicle_capacity) # set up the vehicles

        # generate events
        self._generate_events() # generate events for NYC taxi data

        # set up metrics
        if self.metric_master is None:
            self.setup_metrics()

        # finalize visualization --> prepare values and then give this to visualization
        if Environment.DO_VISUALIZATION:
            # prepare values
            vehicles = list(self._vehicles.values())
            vehicle_positions = {
                vehicle.id: self._geo.get_location_coordinates(vehicle.anchor_location)
                for vehicle in vehicles
            }
            vehicle_occupancies = {
                vehicle.id: len(vehicle.passengers)
                for vehicle in vehicles
                }
            vehicle_max_occupancies = {
                vehicle.id: vehicle.max_passengers
                for vehicle in vehicles
            }
            coordinates = {
                location: self._geo.get_location_coordinates(location=location)
                for location in list(self._geo.get_location_ids())
            }
            
            self._visualization.finalize_init(
                coordinates=coordinates,
                func_plotbackground=self._geo.plot_geography_map,
                limits_coordinates=self._geo.get_limits_location_coordinates(),
                vehicle_positions=vehicle_positions,
                vehicle_occupancies=vehicle_occupancies,
                vehicle_max_occupancies=vehicle_max_occupancies,
                waiting_passengers=self._waiting_passengers.return_count_only())

        if ManGrid.DEBUG_PRINT:
            ManGrid._debug_print("DONE with setting up ManGrid Environment.")
        if Environment.SAVE_ENVIRONMENT:
            self.save_environment()
    
    def re_init(self, reset_vehicles: bool = False,num_vehicles: int = 5,vehicle_capacity: int = 4):
        if reset_vehicles:
            self._vehicles = dict()
            self._setup_vehicles(num_vehicles=num_vehicles,
                             vehicle_capacity=vehicle_capacity)
            self.metric_master = None # reset metric master, so it is set up again
            self.setup_metrics()
        if Environment.DO_VISUALIZATION:
            self._visualization: EnvironmentVisualization \
                = EnvironmentVisualization(self._event_queue)
            # prepare values
            vehicles = list(self._vehicles.values())
            vehicle_positions = {
                vehicle.id: self._geo.get_location_coordinates(vehicle.anchor_location)
                for vehicle in vehicles
            }
            vehicle_occupancies = {
                vehicle.id: len(vehicle.passengers)
                for vehicle in vehicles
                }
            vehicle_max_occupancies = {
                vehicle.id: vehicle.max_passengers
                for vehicle in vehicles
            }
            coordinates = {
                location: self._geo.get_location_coordinates(location=location)
                for location in list(self._geo.get_location_ids())
            }

            # self._visualization._func_plotbackground(gca())
            self._visualization.finalize_init(
                coordinates=coordinates,
                func_plotbackground=self._geo.plot_geography_map,
                limits_coordinates=self._geo.get_limits_location_coordinates(),
                vehicle_positions=vehicle_positions,
                vehicle_occupancies=vehicle_occupancies,
                vehicle_max_occupancies=vehicle_max_occupancies,
                waiting_passengers=self._waiting_passengers.return_count_only())
        if ManGrid.DEBUG_PRINT:
            ManGrid._debug_print("DONE with setting up ManGrid Environment.")


    def get_location_ids(self):
        return self._geo.get_location_ids()

    def get_geography(self):
        """
        Returns the geography object of this environment
        """
        return self._geo
    def get_vehicle_location(self,vehicle_id: uuid.UUID) -> uuid.UUID:
        """
        Returns the current location of the vehicle

        Note: Used in some controllers, the function basically returns the anchor
        """
        if vehicle_id not in self._vehicles:
            raise ValueError(f"Vehicle with id {vehicle_id} does not exist.")
        return self._vehicles[vehicle_id].get_vehicle_location()
    ### PRIVATE ###

    def _callback_next_event(self, event: Event):
        return
    def _callback_register_action(self, action: Action):
        return
    
    def _callback_visualization_update(self,
                                      sim_time: SimTime):
        # prepare values
        vehicle_positions = {
            vehicle_id: self._get_vehicle_coordinates(
                timepoint_eval=sim_time,
                vehicle_id=vehicle_id
            )
            for vehicle_id in self._vehicles
        }
        vehicle_occupancies = {
            vehcile_id: len(vehicle.passengers)
            for vehcile_id, vehicle in self._vehicles.items()
        }

        # send update to visualization
        self._visualization.update(sim_time=sim_time,
                                   vehicle_positions=vehicle_positions,
                                   vehicle_occupancies=vehicle_occupancies,
                                   waiting_passengers=self._waiting_passengers.return_count_only())

    def _generate_events(self):
        """
        Generates events for New York City taxi data
        """
        data = pd.read_csv('code/trip_data_050513_110513.csv')
        startTime = dt.datetime.strptime('2013-05-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        
        locations = np.zeros((data.shape[0], 2), dtype=uuid.UUID)
        request_times = np.zeros((data.shape[0]), dtype=float)
        for idx, row in data.iterrows():
            lat_p = row['pickup_latitude']
            long_p = row['pickup_longitude']
            lat_d = row['dropoff_latitude']
            long_d = row['dropoff_longitude']
            pickup_id,pickup_dist = self._geo.get_nearest_node(lat_p,long_p)
            dropoff_id,dropoff_dist = self._geo.get_nearest_node(lat_d,long_d)
            if pickup_dist > 100 or dropoff_dist > 100:
                # if the distance is too far (assume it is not in manhattan), skip this request
                continue
            if pickup_id == dropoff_id:
                # if pickup and dropoff are the same, skip this request
                continue
            locations[idx, 0] = pickup_id
            locations[idx, 1] = dropoff_id
            travel_duration = self._geo.predict_travel_duration(
                departure_location=pickup_id,
                arrival_location=dropoff_id
            )
            requestTime = pd.to_datetime(row['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
            requestTime = (requestTime - startTime).total_seconds()
            request_times[idx] = requestTime
        # put stuff into the queue
        for i in range(request_times.shape[0]):
            #extract values
            if locations[i,0] == 0 or locations[i,1] == 0:
                # if there is no pickup or dropoff location, skip this request
                continue
            passenger_id = uuid.uuid4()
            location_pickup = locations[i,0]
            location_dropoff = locations[i,1]
            travel_duration = self._geo.predict_travel_duration(departure_location=location_pickup,
                                                                arrival_location=location_dropoff)
            timepoint = SimTime(request_times[i])
            #add passenger to environment
            self._passengers[passenger_id] = Passenger(
                id = passenger_id,
                location_droppoff=location_dropoff,
                location_pickup=location_pickup,
                expected_travel_time=travel_duration
            )
            #put riderequest into queue
            self._event_queue.put(RideRequest(
                timepoint=timepoint,
                location_pickup=location_pickup,
                location_droppoff=location_dropoff,
                passenger_id=passenger_id
            ))

        if ManGrid.DEBUG_PRINT:
            ManGrid._debug_print(f"Generated {request_times.shape[0]} pickup requests in the event queue.")

    def _setup_vehicles(self,
                        num_vehicles: int, #number of vehicles that should be spawned
                        vehicle_capacity: int #capacity per vehicle
                        ):
        """
        Setting up the vehicles in the environment

        How:
        - constant capacity
        - random location among grid
        """
        location_ids = list(self._geo.get_location_ids())
        random_ids = self._rng.integers(0, len(location_ids), num_vehicles)
        initial_locations = [location_ids[i] for i in random_ids]

        for i in range(num_vehicles):
            id = uuid.uuid4()
            self._vehicles[id] = Vehicle(
                    id=id,
                    max_passengers=vehicle_capacity,
                    initial_location=initial_locations[i]
                )
            self._vehicle_events[id] = [] # initialize empty list for vehicle events # NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
            
    def _get_vehicle_coordinates(self,
                                 timepoint_eval: SimTime,
                                 vehicle_id: uuid.UUID) -> np.ndarray:
        """
        Obtain the coordinates for a given vehicle at a future times.
        Uses Geography object to do so.

        If vehicle is idling, send coordinates of idling point.
        If it is driving, assume that it dirves through entire vehicle trajectory.

        NOTE: In case it is ever implemented, that vehicles can wait until a certain
              time at a location and then continue, this needs to be changed!

        Parameters
        ----------
        timepoint_eval: SimTime
            timepoint at which the coordinates should be returned = evaluation timepoint

        Returns
        -------
        np.ndarray: [x,y] where x,y are the coordinates (in m) of the vehicle at the evaluation time
        """

        vehicle = self._vehicles[vehicle_id]
        if vehicle.state == VehicleState.IDLING:
            return self._geo.get_location_coordinates(
                location=vehicle.anchor_location,
                do_assert_location=False
            )
        
        return self._geo.get_vehicle_coordinates_approx(
            timepoint_eval=timepoint_eval,
            timepoint_anchor=vehicle.anchor_timepoint,
            location_anchor=vehicle.anchor_location,
            vehicle_traj=vehicle.traj

        )  
            
    def save_environment(self,filename: str = "ManGrid_env.pkl"):
        with open(filename, 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)

    ### DEBUG ###
    def _debug_print(str: str):
        print("[ManGrid] "+str)
    
    def get_geography(self):
        """
        Returns the geography object of this environment
        """
        return self._geo