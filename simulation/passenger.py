import uuid
from simetime import SimTime, SimDuration
from typing import Callable

from simetime import SimTime

class Passenger:
    """
    Passenger in the system that has a certain ride-request

    Attributes
    ----------
    id: uuid.UUID
        ID of passenger
    location_pickup: uuid.UUID
        where passenger wants to be picked up
    location_dropoff: uuid.UUID
        where passanger wants to be dropped off
    time_spawn: SimTime
        spawn time of passenger = when request came in; None if unkown
    time_pickup: SimTime
        time when passenger is picked up; None if unkown
    time_droppoff: SimTime
        time when passenger is dropped off; None if unkown
    """

    def __init__(self,
                 id: uuid.UUID,
                 location_pickup: uuid.UUID, #pickup and dropoff location
                 location_droppoff: uuid.UUID,
                 expected_travel_time: SimDuration):
        # assign properties
        self.id: uuid.UUID = id #assign id
    
        self.location_pickup: uuid.UUID = location_pickup
        self.location_dropoff: uuid.UUID = location_droppoff

        self.time_spawn: SimTime = None
        self.time_pickup: SimTime = None
        self.time_dropoff: SimTime = None
        self.expected_travel_time:SimDuration = expected_travel_time
