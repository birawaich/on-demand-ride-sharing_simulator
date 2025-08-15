from enum import IntEnum
import uuid
import warnings
from dataclasses import dataclass
import copy
from typing import Callable

from passenger import Passenger
from simetime import SimTime, SimDuration
from event import VehicleArrival


class VehicleState(IntEnum):
    """
    Enumerated Type to describe the state of the vehicle

    Attributes
    ----------
    IDLING
        Vehicle is idling = not moving. Stays by convetion at first entry of trajectory
    DRIVING
        Vehicle is driving. Is by convention somewhere between first and second entry of trajectory
    """
    IDLING = 0
    DRIVING = 1

@dataclass
class VehiclePlanEntry:
    """
    Entry of a vehicle plan
    
    Attributes
    ----------
    location: uuid.UUID
        Location this entry refers too
    pasenger_ids: set[uuid.UUID]
        Set of passengers that ought to be picked up at this location
    
    """
    location: uuid.UUID
    passenger_ids: set[uuid.UUID]

class VehiclePlan:
    """
    Store plan of a vehicle i.e. where to go and who to pick up there

    Attributes
    ----------
    _entries: list[VehiclePlanEntry]
        (private) entries of the vehicle plan; order of list is order in which
        vehicle plans to drive
    """

    def __init__(self,
                 entries: list[VehiclePlanEntry]= []):
        self._entries: list[VehiclePlanEntry] = copy.deepcopy(entries) #this needs to be copied, otherwise they all have the same entry list

    def __deepcopy__(self, memo):
        copied_entries = copy.deepcopy(self._entries, memo)
        return VehiclePlan(copied_entries)
    
    def __str__(self):
        return "\n".join(f"{i}: {str(obj)}" for i, obj in enumerate(self._entries))
    def __repr__(self):
        return str(self)
    
    def get_passengers_at_location(self,
                                   location: uuid.UUID) -> set[uuid.UUID]:
        """Returns the set of passengers that ought to be picked up at given location"""
        for entry in self._entries:
            if entry.location == location:
                return entry.passenger_ids
        return []
    
    def get_plan_locations(self) -> list[uuid.UUID]:
        """Returns a list of only the locations"""
        return [entry.location for entry in self._entries]
    
    def get_next_location(self) -> uuid.UUID:
        """Return only next location, None if thee is none"""
        if len(self._entries) == 0:
            return None
        return self._entries[0].location
    
    def advance_plan(self) -> set[uuid.UUID]:
        """advanced plan by one step i.e. removes the first entry if present
        
        return the passengers that should be picked up there"""
        if len(self._entries) == 0:
            return set()
        # debugList = []
        # for entry in self._entries:
        #     if not debugList.__contains__(entry.passenger_ids):
        #         debugList.append(entry.passenger_ids)
        #     # debugList = list(set(debugList)) # make sure it is a set
        #     if len(debugList) > 3:
        #         warnings.warn(f"VehiclePlan has more than 3 entries with passengers: {debugList}")
        passengers = self._entries[0].passenger_ids

        del self._entries[0]
        return passengers
    
    def squish_plan(self):
        """
        Squish plan by combining entries that have the same location that are back to back
        """
        if len(self._entries) < 2:
            return
        i = 0
        while(1):
            if self._entries[i].location == self._entries[i+1].location:
                # combine entries
                self._entries[i].passenger_ids = \
                    self._entries[i].passenger_ids | self._entries[i+1].passenger_ids
                del self._entries[i+1]
            elif self._entries[i+1].passenger_ids == set():
                del self._entries[i+1] # remove empty entries
            else:
                i += 1
            if i >= len(self._entries) - 1: # if we are at the end, stop
                break

    def get_all_passengers(self) -> set[uuid.UUID]:
        """Return all passengers ID present in the vehicle plan"""
        res = set()
        for entry in self._entries:
            res = res | entry.passenger_ids
        return res
    
    def extend(self,
               entries: list[VehiclePlanEntry],
               from_beginning: bool = False):
        """Extend the vehicle plan with the following entries at the end
        
        can flip flag to insert from the left"""
        if from_beginning:
            self._entries = entries + self._entries
        else:
            self._entries.extend(entries)

    def append(self,
               entry: VehiclePlanEntry):
        """Append a single entry to the vehicle plan"""
        self._entries.append(entry)
    
    def remove(self,
               entry: VehiclePlanEntry):
        self._entries.remove(entry)

    def insert(self,
               index: int,
               entry: VehiclePlanEntry):
        """Insert a single entry at the beginning of the vehicle plan"""
        self._entries.insert(index, entry)

    def modify(self, entries: list[VehiclePlanEntry]):
        """
        Modify the vehicle plan with the following entries, replacing the current entries
        """
        self._entries = copy.deepcopy(entries)

class VehicleTrajectory:
    """
    Class storing what locations the vehicle will be at in the future

    Note: compared to the vehicle plan, this also stores the location where nothing happens

    Attributes
    ----------
    _traj: list[tuple[uuid.UUID,SimTime]]
        (private) Trajectory which locations the vehicle will traverse at which time

        _Note_: Relation to Plan
        The trajectory is all the locations in order to achieve the plan, whereas 
        the plan is only the points at which passengers will be picked up or dropped off
    """

    def __init__(self):
        self._traj: list[tuple[uuid.UUID,SimTime]] = [] #empty traj

    def __str__(self):
        return "\n".join(f"{i}: {str(obj)}" for i, obj in enumerate(self._traj))
    def __repr__(self):
        return str(self)

    def update(self,
               traj: list[(uuid.UUID, SimDuration)],
               start_time: SimTime):
        """
        Update the trajectory of the vehicle'

        Parameters
        ----------
        traj: list[(uuid.UUID, SimDuration)]
            New Trajectory that will be added, where each item tells how long it 
            takes to reach this point from the previous point
        start_time: SimTime
            Start time when vehicle starts driving at the first location
        """
        self._traj = traj #note: her the attribute has Durations and not Times in the 2nd field. This is overwritten further down
        if len(traj) == 0: #if no trajectory needs to be added, return
            return
        
        self._traj[0] = (self._traj[0][0],copy.copy(start_time)) #set the first entry to the start time
        for i in range(1,len(self._traj)):
            # set the time of the trajectory points to be relative to the start time
            trajpt = self._traj[i]
            self._traj[i] = (trajpt[0],
                             SimTime(trajpt[1].duration_s + self._traj[i-1][1].time_s))
        return
    
    def get_closest_point_in_time(self,
                                  time: SimTime,
                                  in_future: bool = True) -> tuple[uuid.UUID, SimTime]:
        """
        Return the next (=future (default) or past) point (a location and a corresponding time)
        given the reference time.

        Parameters
        ----------
        time: SimTime
            time that is refered to
        in_futre: bool
            true (default) to refer to the future, false to refer to the past

        Returns
        -------
        Found tuple of location and time in trajectory, a tuple of none if nothing found
        """
        if in_future:
            return next((trajpt for trajpt in self._traj if trajpt[1] >= time)
                        ,(None,None))
        else:
            last = (None, None)
            for trajpt in self._traj:
                if trajpt[1] >= time:
                    break
                last = trajpt
            return last

    def get_time_of_arrival(self,
                            location: uuid.UUID) -> SimTime:
        """
        Obtain the time when a vehicle will be at a certain location

        Parameters
        ----------
        location: uuid.UUID
            desired location (must be valid and within the trajectory)
        
        Returns
        -------
        SimTime when there. None if never reaching
        """
        point = next((trajpt for trajpt in self._traj if trajpt[0] == location),
                     (None,None))
        return point[1]
    
    def get_time_range(self) -> tuple[SimTime,SimTime]:
        """
        Return the minimal and maximal time present in the trajectory.
        None if the the trajectory is empty
        """
        if len(self._traj) == 0:
            return (None,None)
        return (self._traj[0][1],self._traj[-1][1])

    def cleanup_traj(self,
                     time:SimTime) -> tuple[uuid.UUID,SimTime]:
        """
        Cleanup the trajectory of the vehicle to only contain entries that 
        are in the future as well as the current location

        Note: should only be called when the anchor is set.

        Parameters
        ----------
        time: SimTime
            current time, cleanup until then

        Returns
        -------
        tuple[uuid.UUID,SimTime]
            Tupel of what is now the first element, None elements if this is empty
        """
        if len(self._traj) == 0:
            return (None,None)

        for i in range(len(self._traj)):
            if self._traj[i][1] >= time:
                self._traj = self._traj[i:] # keep the rest of the trajectory
                break # as the trajectory is ordered, we can stop here
            if i == len(self._traj)-1:
                self._traj = []
                return (None,None)
        return self._traj[0]
        
    def get_num_entries(self) -> int:
        """return the amount of entries in the trajectory"""
        return len(self._traj)

class Vehicle:
    """
    A vehicle (think shared taxi/bus) in the system.

    Attributes
    ----------

    id: uuid.UUID
        ID of the vehicle
    max_passengers: int
        maximal amount of passengers in vehicle
    passengers: dict[uuid.UUID, Passenger]
        dict of all passengers currently in the vehicle
    state: VehicleState
        state of the vehicle i.e. driving or idling
    plan: VehiclePlan
        future plan of where vehicle should travel andtime pick up whom
    traj: VehicleTrajectory
        full location trajectory = path of locations that the vehicle will cross (only makes sense if vehilce is driving)
    anchor_location: uuid.UUID
        location of where vehicle is (idling) or driving from
    anchor_timepoint: SimTime
        timepoint of when vehicle swapped to current state at Anchor
    next_arrival_event: VehicleArrival
        next arrival event of this vehicle, none if there is not any

    """

    def __init__(self,
                 id: uuid.UUID,
                 max_passengers: int,
                 initial_location: uuid.UUID):
        # assign properties
        self.id: uuid.UUID = id
        self.max_passengers: int = max_passengers

        # fill data structures
        self.passengers: dict[uuid.UUID, Passenger] = dict()
        self.assigned_passengers: int = 0
        self.assigned_passengers_list: list[uuid.UUID] = [] # list of assigned passengers

#         #DEBUG
#         self.dropped:int = 0
#         self.next_event: Event = None # next event time, used for debugging
#         self.old_traj: list[tuple[uuid.UUID,SimTime]] = [] # old trajectory, used for debugging
#         self.last_clean_time: SimTime = SimTime(0.0) # last time the trajectory was cleaned up, used for debugging
#         # note: do not want to use set, as with dict we can find the passenger object by ID and then return it
# >>>>>>> novel_controller

        self.anchor_location = initial_location
        self.anchor_timepoint: SimTime = SimTime(0.0)
        self.state: VehicleState = VehicleState.IDLING #cars start idling
        self.plan: VehiclePlan = VehiclePlan() #empty plan
        self.traj: VehicleTrajectory = VehicleTrajectory() #empty trajectory

        self.next_arrival_event: VehicleArrival = None

    def arrive_at_location(self,
                           location: uuid.UUID,
                           arrival_time: SimTime) -> set[uuid.UUID]:
        """
        Arrive at a specific location, updates internal states

        -   cleans the trajectory
        -   asserts that arrving according to plan
        -   removes last anchor
        -   update anchor time
        -   sets the car to IDLING

        Parameters
        ----------
        location: uuid.UUID
            Location where the vehicle should arrive at
        arrival_time: SimTime
            Time when arrival happens

        Returns
        -------
        Passengers that are supposed to be picked up at arrival location according to plan
        """
        first_location,_ = self.traj.cleanup_traj(arrival_time) # cleanup the trajectory to only contain entries that are in the future/current location
        assert first_location == location, \
            f"Vehicle arrived at location which is not next trajectory point after cleanup, THIS SHOULD NOT OCCUR! clean up your cleanup function!"
        
        next_location = self.plan.get_next_location()
        assert next_location == location, \
            f"Vehicle arrived at location {location}" \
            +f"yet according to the vehicle plan it should go to {next_location}.\n"

        self.anchor_location = location
        self.anchor_timepoint = copy.copy(arrival_time)

        self.state = VehicleState.IDLING
        passengers = self.plan.advance_plan() # get the passengers that should be picked up at this location
        pass_copy = copy.deepcopy(passengers)
        for p in passengers:
            if p in self.passengers.keys():
                # print(f"Passenger already on board")
                pass_copy.remove(p)
            if p not in self.assigned_passengers_list:
                print(f"Vehicle {self.id} tried to pick up passenger {p} that is not assigned to it. This should not happen!")
        #         passengers.remove(p) # remove passengers that are not assigned to this vehicle
        return pass_copy
            
    def dropoff_passengers(self,
                           location: uuid.UUID,
                           time: SimTime) -> list[uuid.UUID]:
        """
        Dropoff all passengers that want to go to the specified location

        Parameters
        ----------
        location: uuid.UUID
            Location where dropoff should be attempted
        time: SimTime
            Time of when the dropoff happened (to register in passenger)
        
        Returns
        -------
        list[uuid.UUID]
            List of passenger IDs that got dropped off, empty if nobody got dropped off
        """
        dropped_off = []
        for passenger in self.passengers.values():
            if passenger.location_dropoff == location:
                dropped_off.append(passenger.id)
                passenger.time_dropoff = SimTime(time.time_s)
        # delete values
        for id  in dropped_off:
            if id not in self.passengers:
                warnings.warn(f"Passenger with ID {id} not in vehicle's passenger list, cannot drop off.")
                continue
            del self.passengers[id]
            self.assigned_passengers_list.remove(id)

        # Remove any entries related to dropped off passengers
        self.assigned_passengers -= len(dropped_off) # NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
        
        return dropped_off
    
    def pickup_passengers(self,
                          passengers: list[Passenger],
                          time: SimTime):
        """
        Pick up the passengers given by the list with the vehicle, sets time of pickup

        Parameters
        ----------
        passenger: list[Passenger]
            List of passengers that should be picked up
            Throws error if list it too large
        time: SimTime
            When pickup happens
        """
        #assert dimensions
        assert len(passengers) <= self.max_passengers - len(self.passengers), \
                f"Want to pick up {len(passengers)} passengers, "\
                +f"but already have {len(self.passengers)} loaded and a "\
                +f"maximal capacity of {self.max_passengers}." #assertion for pickup size
        #set time and pick up
        for passenger in passengers:
            passenger.time_pickup = SimTime(time.time_s) #set pickup time
            self.passengers[passenger.id] = passenger
    
    def update_vehicle_plan(self,
                            vehicle_plan: VehiclePlan):
        """
        Update the vehicle plan by doing a deepcopy as given. Assumes that it is correct.

        Note: does a deepcopy to not have unwanted sideeffects by the objects 
              in the action being altered by the controller

        Parameters
        ----------
        vehicle_plan: VehiclePlan
            new vehicle plan
        """
        copied_vehicleplan = copy.deepcopy(vehicle_plan)
        self.plan = copied_vehicleplan
    
    def update_vehicle_traj(self,
                            func_get_traj: Callable[[list[uuid.UUID]],None],
                            time: SimTime = None):
        """
        Update the trajectory based on the stored vehicle plan 
        and the anchor time (= assume starting to drive there)

        To enable hotswap updates a time can be passed from which the trajectory
        should be updated

        Note: a function handle is passed s.t. there are no circulat imports as
              the geography class needs to know about Vehicle Trajectories too

        Parameters
        ----------
        func_get_traj: Callable[[list[uuid.UUID]],None]
            geography class to calculate the reference trajectory
        time: SimTime
            (optinal) time to update trajectory from, usueful if
            part of the trajecotry should be kept e.g. during hotswap
        """
        if time is None:
            time = self.anchor_timepoint

        arrival_location, arrival_time = self.get_next_location_in_traj(time)
        if arrival_location is None:
            arrival_location = self.anchor_location
            arrival_time = self.anchor_timepoint

        location_ids = [arrival_location] + self.plan.get_plan_locations()
        
        new_traj = func_get_traj(location_ids)
        self.traj.update(new_traj,arrival_time)
    
    def get_next_location_in_traj(self,
                                  time: SimTime) -> tuple[uuid.UUID,SimTime]: 
        """
        Get the current location based on the trajectory of the vehicle at a specific time

        If the vehicle is idling, this is the anchor time.
        If the trajectory is empty, the anchor point and the current time is returned.
        If in this sime time nothing is reached, a None obj is passed

        Note: in the original implementation this also did a cleanup of the trajectory. 
        This is not perserved as with this new implementation it can also be called when 
        the time is not the latest simtime

        Parameters
        ----------
        time: SimTime
            time of interest

        Returns
        -------
        next location and the time when there as a tuple
        """
        if self.state == VehicleState.IDLING \
            or self.traj.get_num_entries() == 0: # if idling or no trajectory, return the anchor location and current time
            return self.anchor_location,time 
        return self.traj.get_closest_point_in_time(time=time)
           
    def update_vehicle_state(self,
                             new_state: VehicleState,
                             sim_time: SimTime) -> uuid.UUID:
        """
        Process vehicle states, potentially adjust anchor time
        and returns data for a new event (if needed)

        Parameters
        ----------
        new_state: VehicleState
            new desired state
        sim_time: SimTime
            time when this state change happened

        Returns
        -------
        location: uuid.UUID
            new location where this vehicle will drive to!
            None if it wont cause a VehicleArrival Event eventually
        """
        old_state = self.state
        if new_state is old_state:
            return None

        res = None
        if old_state is VehicleState.IDLING and new_state is VehicleState.DRIVING:
            # IDLING --> DRIVING
            # print(f"Vehicle {self.id} is driving")
            self.anchor_timepoint = copy.copy(sim_time)
            res = self.plan.get_next_location()
        else:
            raise NotImplementedError(f"A transition from {old_state} to {new_state} is "+
                                        f"NOT implemented. Before implementing it, think if it should be "+
                                        "implemnted or if the validation needs to be extended.")

        # set new state
        self.state = new_state

        return res
    
    ### QUESTIONABLE FUNCTIONS

    def get_vehicle_location(self) -> uuid.UUID:
        """
        NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
        Get the current location of the vehicle
        Note: Just returns anchor! Does not make use of trajectories.

        Returns
        -------
        uuid.UUID
            ID of the current location of the vehicle
        """
        return self.anchor_location
    def get_traj(self,
            time: bool =True) -> list[(uuid.UUID, SimTime)]:
        """
        NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV

        Since a questionable function it is not super beautiful and just uses private fiels of trajectory.
        Note: state of the vehicle is ignored, this function is only valid if the vehicle is driving.

        Return the trajectory of the vehicle as a list of tuples (location_id, time)
        time=True: return the trajectory with time, False: return the trajectory without time 
        """
        if self.state != VehicleState.DRIVING:
            warnings.warn("Returing a trajectory for an idling vehicle. This is most definitly wrong.")

        if self.traj.get_num_entries() == 0:
            return [self.anchor_location] # if no trajectory, return the anchor location as a list
        if time:
            return copy.deepcopy(self.traj._traj)
        else:
            return [trajpt[0] for trajpt in self.traj._traj] # return the trajectory without time, just the location IDs
    def assign_passenger(self,passenger:uuid.UUID): # NOT RELEVANT FOR CORE FUNCTIONALITY OF ENV
        self.assigned_passengers += 1
        self.assigned_passengers_list.append(passenger)

