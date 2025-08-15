import uuid
from typing import KeysView
import warnings

from vehicle import VehicleState, VehiclePlan


class Action:
    """
    Action = Way of Controller/Agent to influence the Evironment

    By convention vehicles not described by the action will retain their current state.
    Hence this action class does _not_ have to describe all vehicles each time it is
    created.

    Note: As there is only one action at at a time, the action class
          encompasses all possible actions i.e. there is no subclasses
          for different actions as with events.

    Attributes
    ----------
    vehicle_plans: dict[uuid.UUID, VehiclePlan]
        Dictionary of (new) vehicle plans
        Vehicle ID --> What this vehicle should do
    vehicle_stae: dict[uuid.UUID, VehicleState]
        Dictionary of (new) vehicle state
        Vehicle State --> In what state this vehicle should be

    _valid_ids: bool
        (private) Whether the used IDs are valid according to values given
    _valid_assignments: bool
        (private) Whether the assignments are valid according to the values given
    """

    def __init__(self,
                 vehicle_plans: dict[uuid.UUID, VehiclePlan] = {},
                 vehicle_states: dict[uuid.UUID, VehicleState] = {}):
        # set attributes
        self.vehicle_plans: dict[uuid.UUID, VehiclePlan] = vehicle_plans
        self.vehicle_states: dict[uuid.UUID, VehicleState] = vehicle_states
        # set defaults
        self._valid_ids = False
        self._valid_assignments = False

    def __str__(self):
        return f"[\n\tVEHICLE PLANS("+\
            "".join(f'Vehicle={k}: {len(v.get_plan_locations())} Destinations' for k, v in self.vehicle_plans.items())+\
            f");\n\tVEHICLE STATES: {self.vehicle_states}]"

    def validate_ids(self,
                     valid_vehicle_ids: KeysView[uuid.UUID],
                     valid_location_ids: KeysView[uuid.UUID],
                     valid_passenger_ids: KeysView[uuid.UUID]) -> bool:
        """
        Validate the IDs given in the action

        -   vehicle IDs
            -   in keys plans
            -   in keys states
        -   location IDs
            -   in values plans
        -   passenger IDs
            -   in values plans <-- note, would not break anything if not valid!

        Parameters
        ----------
        valid_vehicle_ids: KeysView[uuid.UUID]
            Key View of valid vehicle IDs
        valid_location_ids: KeysView[uuid.UUID]
            Key View of valid location IDs
        valid_passenger_ids: KeysView[uuid.UUID]
            Key View of valid passenger IDs

        Returns
        -------
        bool: True if valid, False otherwise
        """

        # vehicle ids
        if not self.vehicle_plans.keys() <= valid_vehicle_ids:
            warnings.warn("Some of the vehicle IDs in the vehicle plans are invalid.")
            self._valid_ids = False
            return False
        if not self.vehicle_states.keys() <= valid_vehicle_ids:
            warnings.warn("Some of the vehicle IDs in the vehicle states are invalid.")
            self._valid_ids = False
            return False

        # location IDs
        used_locations = set()
        for vehicle_plan in self.vehicle_plans.values():
            used_locations = used_locations | set(vehicle_plan.get_plan_locations())
        if not used_locations <= valid_location_ids:
            warnings.warn("Some of the location IDs in the vehicle plans are invalid.")
            self._valid_ids = False
            return False

        # passenger IDs
        used_passengers = set()
        for vehicle_plan in self.vehicle_plans.values():
            used_passengers = used_passengers | vehicle_plan.get_all_passengers()
        if not used_passengers <= valid_passenger_ids:
            warnings.warn("Some of the passesnger IDs in the vehciles plans are invalid")
            self._valid_ids = False
            return False

        self._valid_ids = True
        return True
    
    def validate_assignemnt(self,
                            driving_vehicles: dict[uuid.UUID, uuid.UUID]) -> bool:
        """
        Validate the Assignment logically. Throws Error if IDs have not been validated.

        -   a vehicle currently driving cannot suddenly idle
            (need to arrive at a location, then idles automatically)
        -   if the vehicle is driving and the plan is changed, the first node in the plan must stay the same
        
        Note: Vehicle Capacity restrictions are checked when arriving  
        This is as currently there is no convenient way to read out which passengers leaves the vehicle 
        where and it is already being done in `Environment.next_event()`. If there is a requirement to 
        filter out faulty assignments here, alter this code!

        Parameters
        ----------
        driving_vehicles: dict[uuid.UUID, uuid.UUID]
            Dictionary of all driving vehicles  
            (driving) Vehcile ID --> First Node on plan

        Returns
        -------
        bool True if the assignments are valid, false otherwise
        """
        if not self._valid_ids:
            raise Exception("IDs must be valid before this function can be called!")
        
        for vehicle_id in driving_vehicles.keys():
            # check if this vehicle is being set to idle
            if vehicle_id in self.vehicle_states\
                and self.vehicle_states[vehicle_id] == VehicleState.IDLING:
                warnings.warn("At least one vehicle currently driving is set to imemdiatly idle. This is not valild. Can only idle at locations.")
                self._valid_assignments = False
                return False
            
            # check if the plan still has the same next node --> thjis should now be allowed!
            # if vehicle_id in self.vehicle_plans\
            #     and driving_vehicles[vehicle_id] != self.vehicle_plans[vehicle_id].get_next_location():
            #     warnings.warn("At least one vehicle currently driving is set to alter the plan to drive to a new immediate destiantion. This is not allowed.")
            #     self._valid_assignments = False
            #     return False

        self._valid_assignments = True
        return True
    
    def find_routehotswapping_vehicles(self,
                                       driving_vehicles: dict[uuid.UUID,uuid.UUID]) -> set[uuid.UUID]:
        """
        Find all vehicles (from the ones currently driving) that are hotswapping the route and return them as a set
        Note: Hotswapping: currently driving to somewhere, but changing this immediate destination during the route!

        Parameters
        ----------
        driving_vehicles: dict[uuid.UUID, uuid.UUID]
            Dictionary of all driving vehicles  
            (driving) Vehcile ID --> First Node on plan

        Returns
        -------
        set of all vehicle IDs that are doing such a hotswap
        """
        if not self._valid_ids and not self._valid_assignments:
            raise Exception("IDs and assigments must be valid before this function can be called!")
        
        res = set()

        for vehicle_id in driving_vehicles.keys():
            # check if the plan still has the same next node --> thjis should now be allowed!
            if vehicle_id in self.vehicle_plans\
                and driving_vehicles[vehicle_id] != self.vehicle_plans[vehicle_id].get_next_location():
                res.add(vehicle_id)
            
        return res