from abc import ABC, abstractmethod
import uuid

from simetime import SimTime

class Event(ABC):
    """
    Event as it happens in the environment.

    Think new ride request

    Attributes
    ----------
    timepoint: SimTime
        point in time when this event has/will happen
    """
    
    @abstractmethod
    def __init__(self,
                 timepoint: SimTime, #time when it happens
        ):
        self.timepoint = timepoint #when this observation occurs (not is observed)


class RideRequest(Event):
    """
    Class to store a Ride Request Event

    Attributes
    ----------
    location_pickup: uuid.UUID
        location of the pickup
    location_dropoff: uuid.UUID
        location of the dropoff
    passsenger_id: uuid.UUID
        ID of the corresponding passenger
    """

    def __init__(self,
                 timepoint: SimTime,
                 location_pickup: uuid.UUID,
                 location_droppoff: uuid.UUID,
                 passenger_id: uuid.UUID):
        super().__init__(timepoint)

        # write properties
        self.location_pickup: uuid.UUID = location_pickup
        self.location_dropoff: uuid.UUID = location_droppoff
        self.passenger_id: uuid.UUID = passenger_id

    def __str__(self):
        return f"[Passenger: {self.passenger_id} @ Pickup: {self.location_pickup} "+\
            f"-> Dropoff: {self.location_dropoff}]"

class VehicleArrival(Event):
    """
    Class to store a vehicle arrived at some location event

    Attributes
    ----------
    location: uuid.UUID
        location where the vehicle arrived
    vehicle_id: uuid.UUID
        ID of the vehicle that arrived
    valid: bool
        true if the event is valid, false if not (ignore in this case)
    """

    def __init__(self,
                 timepoint: SimTime,
                 location: uuid.UUID,
                 vehicle_id: uuid.UUID):
        super().__init__(timepoint)

        # write properties
        self.location: uuid.UUID = location
        self.vehicle_id: uuid.UUID = vehicle_id
        self.valid: bool = True

    def __str__(self):
        return f"[Vehicle: {self.vehicle_id} @ Location: {self.location}]"

class PickupDropoff(Event):
    """
    Class to store a successful Pickup/Dropoff Event

    Attributes
    ----------
    location: uuid.UUID
        location where the pickup happened
    pickup_passenger_ids: list[uuid.UUID]
        IDs of the passengers that got picked up;
        empty if nobody got picked up
    dropoff_passenger_ids: list[uuid.UUID]
        IDs of the passengers that got dropped off;
        empty if nobody got dropped of
    vehicle_id: uuid.UUID
        ID of the vehicle that did the pickup/droppoff
    """

    def __init__(self, timepoint,
                 location: uuid.UUID,
                 pickup_passenger_ids: list[uuid.UUID],
                 dropoff_passenger_ids: list[uuid.UUID],
                 vehicle_id: uuid.UUID):
        super().__init__(timepoint)

        # write properties
        self.location = location
        self.pickup_passenger_ids = list(pickup_passenger_ids)
        self.dropoff_passenger_ids = list(dropoff_passenger_ids)
        self.vehicle_id = vehicle_id

    def __str__(self):
        return f"[Vehicle: {self.vehicle_id} @ Location: {self.location}: "+\
            f"Off: {len(self.dropoff_passenger_ids)}, On: {len(self.pickup_passenger_ids)}]"
    
class VisualizationRequest(Event):
    """
    Class to signal an environment visualizaiton request
    """

    def __init__(self, timepoint):
        super().__init__(timepoint)

class ControllerTimingTick(Event):
    """
    Class that is a timing tick as requested by soome controller

    Attributes
    ----------
    identification: uuid.UUID
        identification ID = set by controller, s.t. it is clear who has to listen
    period_s: float
        period (constant) between ticks in seconds
    """

    def __init__(self, timepoint,
                 identification: uuid.UUID,
                 time_period: float):
        super().__init__(timepoint)

        self.identification: uuid.UUID = identification
        self.period_s: float = time_period