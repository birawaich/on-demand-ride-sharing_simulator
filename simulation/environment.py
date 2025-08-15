from abc import ABC, abstractmethod
import uuid
import warnings
import copy

from event import Event, RideRequest, VehicleArrival, PickupDropoff, VisualizationRequest, ControllerTimingTick
from eventqueue import EventQueue
from action import Action
from passenger import Passenger
from vehicle import Vehicle, VehicleState
from simetime import SimTime, SimDuration
from visualization import EnvironmentVisualization
from geo import Geography
from metrics import MetricMaster

class Environment(ABC):
    """
    Abstract Environment Class

    Defines Interface of the environment.

    Attributes
    ----------

    sim_duration_s
        How long the simualtion should last in simulated seconds
    metric_master: MetricMaster
        Object to handle metrics

    _geo: Geography
        Reference to the gegraphy
    _event_queue: EventQueue
        (private) Event queue that dictates the event driven simulation
    _passengers: dict[uuid.UUID, Passenger]
        (private) Holding all passengers in the environment
    _waiting_passengers: WaitingPassengersIndexget_location_ids
        (private) Index to keep track of who is waiting where in what order
    _vehicles: dict[uuid.UUID, Vehicle]
        (private) Holding all vehicles in the environment
    _lastest_time: SimTime
        (private) Lastest Time of simulation i.e. timepoint of last event; starts at 0
    _visualization: EnvironmentVisualization
        (private) Object to do environment visualization    
    """

    DEBUG_PRINT = False
    """Class Property: set to true for verbose prints"""

    DO_PLOT = True
    """Class Property: Do a plot of the environment metrics after the run."""

    DO_VISUALIZATION = False
    """Class Property: set to true to do a visualization (meaning collecting the data)"""

    SAVE_ENVIRONMENT = False
    """Class Property: set to true to save the environment before the simulation run"""

    LOAD_ENVIRONMENT = False
    """Class Property: set to true to load the environment before the simulation run."""

    @abstractmethod
    def __init__(self,
                 sim_duration_s=24*60*60):
        # set object properties
        self.sim_duration_s = sim_duration_s #duration of the simulation in s, defauts to 1d

        # init event queue
        self._event_queue: EventQueue = EventQueue()

        # state properties
        self._passengers: dict[uuid.UUID, Passenger] = dict() #dictionary of all passengers
        self._vehicles: dict[uuid.UUID, Vehicle] = dict() #dictionary of all vehicles
        self._vehicle_events: dict[uuid.UUID, list[Event]] = dict() #dictionary of all vehicle events # NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV -- #TODO what is this even used for?
        self._waiting_passengers: WaitingPassengersIndex = WaitingPassengersIndex() #waiting passengers
        self._latest_time: SimTime = SimTime(0.0)
        self._geo = self.get_geography()  # geography of the environmen

        #pseudo for metrics --> set up in environment once all is set
        self.metric_master: MetricMaster = None

        # visualization
        if Environment.DO_VISUALIZATION:
            self._visualization: EnvironmentVisualization \
                = EnvironmentVisualization(self._event_queue)
            
    def finalize(self):
        if Environment.DEBUG_PRINT:
            Environment._debug_print("In Environment Finalizer...")

        # evalute matrics
        print(f"=======\nCollected Environment Metrics:\n{self.metric_master}=======")
        if Environment.DO_PLOT:
            self.metric_master.plot()
        
        # visualization
        if Environment.DO_VISUALIZATION:
            self._visualization.create_animation()

    def setup_metrics(self):
        """Setup the metrics, needs to be called AFTER initializations as the vehicle IDs need to be known."""
        self.metric_master: MetricMaster = MetricMaster(
            geo=self._geo,
            vehicle_ids=self.get_vehicles_ids())

    def next_event(self) -> Event:
        """
        Asked the environment for the next event.
        Returns None if the queue is empty.
        Updates latest time.

        Internally processes events to keep the state of the system valid:
        -   `RideRequest`
            -   set passenger spawn time
            -   make passenger waiting at location
        -   Vehicle Arrival
            -   update vehicle state to
                -   new location
                -   idling
            -   try to dropoff passengers
            -   try to pickup passengers  
            Generate a corresponding `PickupDropoff` Event
        -   PickupDropoff  
            Error as this should not be in the queue.

        Returns
        -------
        Event as it should be processed externally = by a controller/agent
        """
        # get event from queue
        event: Event = self._event_queue.get()
        if event is None:
            if Environment.DEBUG_PRINT:
                Environment._debug_print("Event Queue is empty.")
            return None
        if Environment.DEBUG_PRINT:
            Environment._debug_print(f"<<<< Extract Event from Queue @ SimTime: {event.timepoint.get_humanreadable()} >>>>")

        # process generally
        if isinstance(event, RideRequest):
            if not self._passengers[event.passenger_id].time_spawn:            
                self._passengers[event.passenger_id].time_spawn = SimTime(event.timepoint.time_s)
                self._waiting_passengers.add_passenger(event.location_pickup,
                                                   passenger_id=event.passenger_id)
            # note waiting passengers change (potential)
                num_passsenger_waiting = self._waiting_passengers\
                    .return_count_only_at_location(location=event.location_pickup)
                self.metric_master.callback_waiting_passengerchange(
                    time=event.timepoint,
                    old_at_location=num_passsenger_waiting-1, #a ride request ALWAYS means one new passenger
                    new_at_location=num_passsenger_waiting
                    )
            if Environment.DEBUG_PRINT:
                Environment._debug_print(f"New RideRequest: {event}")

        elif isinstance(event, VehicleArrival):
            if Environment.DEBUG_PRINT:
                Environment._debug_print(f"New VehicleArrival: {event}")

            # if the event is invalid: go to the next event
            if not event.valid:
                if Environment.DEBUG_PRINT:
                    Environment._debug_print(f"VehicleArrival event was invalid. Will go to next event.")
                self._latest_time = copy.copy(event.timepoint)
                return self.next_event()

            vehicle = self._vehicles[event.vehicle_id]
            old_state = vehicle.state
            old_passenger_num = len(vehicle.passengers)
            old_passenger_waiting = self._waiting_passengers\
                .return_count_only_at_location(location=event.location)

            # vehicle_copy = copy.deepcopy(vehicle) #copy vehicle for debug print
            location = event.location
            old_vehicle = copy.deepcopy(vehicle) #copy vehicle for debug print
            # update vehicle state
            passenger_ids_topickup = vehicle.arrive_at_location(location=location,
                                                                arrival_time=event.timepoint)
            # try dropping off
            dropped_off = vehicle.dropoff_passengers(location=location,
                                                     time=event.timepoint)

            # try picking up
            assert len(passenger_ids_topickup) <= vehicle.max_passengers - len(vehicle.passengers), \
                f"Want to pick up {len(passenger_ids_topickup)} passengers, "\
                +f"but already have {len(vehicle.passengers)} loaded and a "\
                +f"maximal capacity of {vehicle.max_passengers}."\
                +f"\nAssigned passengers"\
                + f"{vehicle.assigned_passengers_list}"\
                +f"\n Picking up: {passenger_ids_topickup}"\
                +f"\n Dropping Off: {dropped_off}"\
                +f"\n On Board: {vehicle.passengers.keys()}"\
                
                      #assertion for pickup size
            passenger_ids_pickedup = self._waiting_passengers.remove_passengers(
                location=location,
                passenger_ids=passenger_ids_topickup) #remove from waiting lsit
            passengers_pickup = [self._passengers[id] for id in passenger_ids_pickedup]
            vehicle.pickup_passengers(passengers=passengers_pickup,
                                      time=event.timepoint) # add passengers to vehicle

            # generate PickupDropoff Event
            event = PickupDropoff(timepoint=copy.copy(event.timepoint),
                                  pickup_passenger_ids=passenger_ids_pickedup,
                                  dropoff_passenger_ids=dropped_off,
                                  location=location,
                                  vehicle_id=vehicle.id)
            if Environment.DEBUG_PRINT:
                Environment._debug_print(f"Resulting in PickupDropoff: {event}")

            # collect measurements
            for id in dropped_off:
                self.metric_master.extract_passenger(self._passengers[id])
            # note state change (potential)
            if vehicle.state != old_state:
                self.metric_master.callback_vehicle_stateupdate(
                    time=event.timepoint,
                    vehicle=vehicle,
                    old_state=old_state
                )
            # note passenger change (potential)
            if len(vehicle.passengers) != old_passenger_num:
                self.metric_master.callback_vehicle_passengerchange(
                    time=event.timepoint,
                    vehicle=vehicle,
                    old_num_passenger = old_passenger_num
                )
            # note waiting passengers change (potential)
            new_passengers_waiting = self._waiting_passengers\
                .return_count_only_at_location(location=location)
            if old_passenger_waiting != new_passengers_waiting:
                self.metric_master.callback_waiting_passengerchange(
                    time=event.timepoint,
                    old_at_location=old_passenger_waiting,
                    new_at_location=new_passengers_waiting
                )

        elif isinstance(event,VisualizationRequest):
            # Environment._debug_print("Add environment visualization.")
            self._callback_visualization_update(sim_time=event.timepoint)
            # potentially add new visualization request to queue
            self._visualization.potentially_add_visualization_request(self._event_queue)

        elif isinstance(event,ControllerTimingTick):
            self._potentially_add_next_controllertimingtick(event)

        elif isinstance(event,PickupDropoff):
            raise TypeError("The event queue should not contain PickupDropff events.")

        else:
            raise NotImplementedError(
                f"No handling of an event of type {type(event)} is implemented!")

        # callback for environment specific processing
        self._callback_next_event(event)

        # returning
        self._latest_time = copy.copy(event.timepoint)
        return event
        
    @abstractmethod
    def _callback_next_event(event: Event):
        """
        Callback function for environment object to process an event 
        after the general processing
        """
        pass

    @abstractmethod
    def get_geography(self) -> Geography:
        """
        USE WITH CAUTION (can potentially screw with environment data)

        Returns the geography of the environment

        Returns
        -------
        Geography: Geography object of the environment e.g. BieleGridGeography
        """
        pass

    def register_action(self,
                        action: Action):
        """
        Register an action to the environment

        -   validate (with action validation functions)
            if invalid --> throw warning and discard it
        -   generate corresponding events and put into Event Queue
        -   update internal states

        Parameters
        ----------
        action: Action
            action (possibly invalid) that should be executed. None for empty action
        """
        if action is None:
            return
        
        if Environment.DEBUG_PRINT:
            Environment._debug_print(f"Register Action: {action}")

        # validate
        are_ids_valid = action.validate_ids(self._vehicles.keys(),
                                            self.get_location_ids(),
                                            self._passengers.keys())
        if not are_ids_valid:
            warnings.warn(f"Discard action due to invalid IDs at time {self._latest_time.get_humanreadable()}.")
            return
        driving_vehicles = dict()
        for id, vehicle in self._vehicles.items():
            if vehicle.state == VehicleState.DRIVING:
                driving_vehicles[id] = vehicle.plan.get_next_location()
        are_assignments_valid = action.validate_assignemnt(driving_vehicles=driving_vehicles)
        if not are_assignments_valid:
            warnings.warn(f"Discard action due to invalid assignments at time {self._latest_time.get_humanreadable()}.")
            return
        
        # find hotswapping vehicles
        hotswapping_vehicles = action.find_routehotswapping_vehicles(driving_vehicles=driving_vehicles)

        # update vehicle plans 
        for id, vehicle_plan in action.vehicle_plans.items():
            self._vehicles[id].update_vehicle_plan(
                vehicle_plan=vehicle_plan
            )

        # process state change --> add events to queue
        for id, new_state in action.vehicle_states.items():
            does_hotswap = id in hotswapping_vehicles
            vehicle = self._vehicles[id]
            old_state = vehicle.state

            next_loc = vehicle.update_vehicle_state(
                new_state=new_state,
                sim_time = self._latest_time
            )
            if next_loc is None and not does_hotswap:
                continue

            # handle hotswap and update vehicle trajectory
            if does_hotswap:
                vehicle.next_arrival_event.valid = False #original arrival event is false
                next_loc = vehicle.plan.get_next_location() #just get the next location where people are
                
                # get the lcoation the vehicle has been previously and set as new anchor
                loc_before, time_before = vehicle.traj.get_closest_point_in_time(time=self._latest_time,
                                                                                 in_future=False)
                if loc_before is not None and time_before is not None:
                    vehicle.anchor_location = loc_before
                    vehicle.anchor_timepoint = time_before

                # udpate the trajectory (with the modified anchor being used)
                vehicle.update_vehicle_traj(func_get_traj=self._geo.get_traj,
                                            time=self._latest_time)
            else:
                vehicle.update_vehicle_traj(func_get_traj=self._geo.get_traj) #no hotswap = anchor got set when updating state

            time_of_arrival = vehicle.traj.get_time_of_arrival(next_loc)
            event = VehicleArrival(
                timepoint=copy.copy(time_of_arrival),
                location=next_loc,
                vehicle_id=id
            )
            vehicle.next_arrival_event = event #link event to vehicle
            self._event_queue.put(event)

            # sanity assertion
            next_loc = vehicle.plan.get_next_location()
            if vehicle.state == VehicleState.DRIVING:
                assert vehicle.traj.get_time_of_arrival(next_loc) == vehicle.next_arrival_event.timepoint\
                    and vehicle.next_arrival_event.timepoint >= self._latest_time, \
                    f"Oh no, the trajectories are wrong: according to the trajectories should "+\
                    f"arrive at next location at {vehicle.traj.get_time_of_arrival(next_loc)} and according "+\
                    f"to next event at {vehicle.next_arrival_event.timepoint}. Further, this timepoint might be in the past. "+\
                    f"(current time: {self._latest_time})"
                # events can be at the same time if the passenger spawns where the vehicle is

            # metrics: callback in case of state change
            if old_state != vehicle.state:
                self.metric_master.callback_vehicle_stateupdate(
                    time=self._latest_time,
                    vehicle=vehicle,
                    old_state=old_state
                )

        # callback
        self._callback_register_action(action=action)

    @abstractmethod
    def _callback_register_action(self,
                                  action: Action):
        """
        Callback function for an envrionment object to process an action
        after the general processing
        """
        pass

    @abstractmethod
    def get_location_ids(self) -> set[uuid.UUID]:
        """
        Return a set of all used location IDs

        Returns
        -------
        set[uuid.UUID] set of all used location IDs
        """
        pass

    def get_vehicles_ids(self) -> set[uuid.UUID]:
        """
        Return a set of all used vehicle IDs

        Returns
        -------
        set[uuid.UUID] set of all used vehicle IDs
        """
        return set(self._vehicles.keys())
    
    @abstractmethod
    def _callback_visualization_update(self,
                                       sim_time: SimTime):
        """
        Callback function for an environment object to do a visualization.

        TODO think why this cannot be in the super class?

        Parameters
        ----------
        sim_time: SimTime
            time during which this is called
        """
        pass

    def get_vehicle(self, uid:uuid.UUID) -> Vehicle:
        """
        NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
        USE WITH CAUTION (can potentially screw with environment data)

        Returns a specific vehicle by ID

        Returns
        -------
        Vehicle: Vehicle object ct
        """
        return self._vehicles[uid]
    def get_passenger(self, uid: uuid.UUID) -> Passenger:
        """
        NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
        USE WITH CAUTION (can potentially screw with environment data)

        Returns a specific passenger by ID
        Parameters
        ----------
        uid: uuid.UUID
            ID of the passenger to return
        Returns
        -------
        Passenger: Passenger object with the given ID
        """
        return self._passengers[uid]

    def insert_timing_events(self,
                             time_period: float,
                             identification: uuid.UUID):
        """
        Insert timing events into the queue seperated by time_duration,
        identified by the identification

        Works by inserting just a single event
        """
        self._event_queue.put(
            ControllerTimingTick(timepoint=SimTime(0.0),
                                 identification=identification,
                                 time_period=time_period)
        )

    def _potentially_add_next_controllertimingtick(self, event: ControllerTimingTick):
        """adds new controller timing ticks of the same sort"""
        # do not add anything if the queue only contains visualization requests
        if self._event_queue.no_action_events_in_queue():
            return
        # add next event
        self._event_queue.put(ControllerTimingTick(
            timepoint=SimTime(event.timepoint.time_s + event.period_s),
            identification=event.identification,
            time_period=event.period_s
        ))

    def passenger_timeout(self,passenger:Passenger,timepoint:SimTime):
        location = passenger.location_pickup
        old_passenger_waiting = self._waiting_passengers\
                .return_count_only_at_location(location=location)
        self._waiting_passengers.remove_passengers(location,set([passenger.id]))
        new_passengers_waiting = self._waiting_passengers\
                .return_count_only_at_location(location=location)
        if old_passenger_waiting != new_passengers_waiting:
                self.metric_master.callback_waiting_passengerchange(
                    time=timepoint,
                    old_at_location=old_passenger_waiting,
                    new_at_location=new_passengers_waiting
                )
    ### DEBUG ###
    def _debug_print(str: str):
        print("[ENV] "+str)


class WaitingPassengersIndex():
    """
    (local) class to keep track of who is waiting at what location

    works by having a dictionary where the values are sets
    """
    def __init__(self):
        self._storage: dict[uuid.UUID, set[uuid.UUID]] = dict()
    
    def add_passenger(self,
                      location: uuid.UUID,
                      passenger_id: uuid.UUID):
        """Add a waiting passenger to location specified by ID"""
        if location not in self._storage:
            self._storage[location] = set()
        self._storage[location].add(passenger_id)

    def remove_passengers(self,
                          location: uuid.UUID,
                          passenger_ids: set[uuid.UUID]) -> set[uuid.UUID]:
        """attempts to remove the set of passengers given from the waiting passengers.
        Returns the truly removed values"""
        # if location is not a key, then certainly nobody is waiting there
        if location not in self._storage:
            return set()
        # else, return the set difference there
        intersection = passenger_ids.intersection(self._storage[location])
        self._storage[location].difference_update(intersection)
        return intersection
    
    def return_count_only(self) -> dict[uuid.UUID,int]:
        """
        Returns a dictionary of only the amount of waiting passengers

        Returns
        -------

        dict[uuid.UUID,int] dictionary mapping location --> amount of waiting passengers
        """
        return {
            id: len(passenger_set)
            for id, passenger_set in self._storage.items()
        }
    
    def return_count_only_at_location(self,
                                      location: uuid.UUID) -> int:
        """Returns the amount of waiting passengers at this location
        
        returns the convention value 0 if the location si not present"""
        return len(self._storage.get(location,[]))