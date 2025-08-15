from abc import ABC, abstractmethod

from event import Event
from action import Action
import uuid
from vehicle import VehiclePlan

class Controller:
    """
    Interface for a controller
    """

    @abstractmethod
    def process_event(self,
                      event: Event) -> Action:
        """
        Processes an avent and generates an action  
        = Control Loop
        """
        raise NotImplementedError("Implement in subclass!")
    @abstractmethod
    def update_vehicle_plans(self,
                             vehicle_plans: dict[uuid.UUID, VehiclePlan]):
        """
        Update the vehicle plan if more than one controller is acting and this controller 
        needs to keep track of vehicle plans
        """
        raise NotImplementedError("Implement in subclass")