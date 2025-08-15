import uuid

from controller import Controller
from event import Event, RideRequest, PickupDropoff, VisualizationRequest
from action import Action
from vehicle import VehiclePlan, VehiclePlanEntry, VehicleState, Vehicle
from bielegrid_geography import BieleGridGeography
from man_geography import ManGridGeography
from bielegrid import BieleGrid
from simetime import SimTime

class MultiplePassengerController(Controller):
    """
    TODO update description, was clearly wrong
    """

    def __init__(self,
                 locations: set[uuid.UUID],
                 vehicles: set[uuid.UUID],
                 geo: BieleGridGeography,
                 env: BieleGrid):
        # assign parameters
        self._location_ids: set[uuid.UUID] = locations
        self._vehicle_ids: set[uuid.UUID] = vehicles
        self._vehicle_ids_ordered: list[uuid.UUID] = list(vehicles)
        self._vehicle_ids_avail: list[uuid.UUID] = list(vehicles) #Store idling vehicles that are going to pick up passengers
        self._vehicle_plans: dict[uuid.UUID, VehiclePlan] = dict()
        self._vehicle_assigned_passengers: dict[uuid.UUID, int] = dict() #Store number of passengers assigned to each vehicle
        self._geo:BieleGridGeography = geo
        self._env:BieleGrid = env
        for id in self._vehicle_ids:
            self._vehicle_plans[id] = VehiclePlan()
            self._vehicle_assigned_passengers[id] = 0 #Initialize number of passengers assigned to each vehicle to 0
        # set up backlog parameters
        self._backlog_requests: list[RideRequest] = [] #backlog of requests that could not be served at their request

    def process_event(self,
                      event: Event) -> Action:
        """
        Processes an avent and generates an action  
        = Control Loop
        """
        action = None

        if event is None:
            return action
        
        if isinstance(event, RideRequest):
            action = self._process_riderequest(event)
        elif isinstance(event, PickupDropoff):
            action = self._process_pickupdropoff(event)
        elif isinstance(event, VisualizationRequest):
            pass #could add in custom controler visualization that is timed with environment
        else:
            raise NotImplementedError(f"The event of type "+
                                      f"{type(event)} is not implemented.")
        
        return action
    
    def _process_riderequest(self,
                             event: RideRequest,
                             specific_vehicle: uuid.UUID = None) -> Action:
        """
        Process RideRequest Event

        Assign Vehicle to Passenger to nearest available vehicle
        Ensure that Driving
        """
        # self._vehicle_ids_avail = [] #Reset available vehicles
        # for vehicle_id in self._vehicle_ids:
        #     vehicle = self._env.get_vehicle(vehicle_id)
        #     if self._vehicle_assigned_passengers[vehicle_id] - vehicle.dropped_passengers < vehicle.max_passengers:
        #         self._vehicle_ids_avail.append(vehicle_id)
        #Check if there are any vehicles available
        if len(self._vehicle_ids_avail) == 0:
            # add to backlog
            # print("No vehicles available, adding to backlog")
            self._backlog_requests.append(event)
            return Action()
        # generate vehicle plan entries
        if event.location_pickup == event.location_dropoff:
            Warning("Pickup and Dropoff location are the same, this is not allowed. Using pickup location for both.")
        entry_pickup = VehiclePlanEntry(
            location=event.location_pickup,
            passenger_ids=set([event.passenger_id]),
        )
        entry_dropoff= VehiclePlanEntry(
            location=event.location_dropoff,
            passenger_ids=set(),
        )
        # entry_dropoff= VehiclePlanEntry(
        #     location=event.location_dropoff,
        #     passenger_ids=set([event.passenger_id]),
        # )

        # Assign to closest taxi/trajectory
        if specific_vehicle is not None:
            responsible_vehicle_id = specific_vehicle
        else:
            responsible_vehicle_id = None
            min_distance = float('inf')
            for free_vehicle in self._vehicle_ids_avail:
                vehicle = self._env.get_vehicle(free_vehicle)
                # vehicle_location = vehicle.get_current_location(event.timepoint)
                # compute distance to pickup location
                if vehicle.state == VehicleState.IDLING:
                    vehicle_traj = [vehicle.anchor_location]
                else:
                    vehicle_traj = vehicle.get_traj(time=False) #Get trajectory of vehicle without time
                taxi_dist = self._compute_min_distance(vehicle_traj, event.location_pickup)
                if responsible_vehicle_id is None or taxi_dist < min_distance:
                    responsible_vehicle_id = free_vehicle
                    min_distance = taxi_dist
        #Create new vehicle plan for responsible vehicle
        #TODO
        curr_plan = self._vehicle_plans[responsible_vehicle_id]
        locations = curr_plan.get_plan_locations()
        shared_pickup = False
        shared_dropoff = False
        i_pickup = 1
        for i in range(len(curr_plan._entries)):
            entry = curr_plan._entries[i]
            if entry.location == event.location_pickup and not shared_pickup:
                entry.passenger_ids.add(event.passenger_id) #Add passenger to pickup entry
                i_pickup = i #Store index of pickup entry
                shared_pickup = True
                break
        for j in range(i_pickup, len(curr_plan._entries)):
            entry = curr_plan._entries[j]
            if entry.location == event.location_dropoff and not shared_dropoff:
                # entry.passenger_ids.add(event.passenger_id)
                shared_dropoff = True
        assigned_vehicle = self._env.get_vehicle(responsible_vehicle_id)
        assigned_vehicle.assign_passenger(event.passenger_id) #Assign passenger to vehicle
        # curr_plan.extend([entry_pickup, entry_dropoff]) #Add pickup and dropoff to plan
        if not shared_pickup: #If pickup is not already in plan, add it
            for entry in curr_plan._entries:
                if self._geo.get_distance(entry.location, event.location_pickup) > self._geo.get_distance(event.location_pickup, curr_plan.get_next_location()):
                    i_pickup = curr_plan._entries.index(entry)
                    break
            curr_plan.insert(1, entry_pickup) #Insert pickup at second position in plan
        if not shared_dropoff: #If dropoff is not already in plan, add it
            curr_plan.append(entry_dropoff) #Append dropoff at end of plan
        # if len(curr_plan._entries) == 0:#If no entries in plan, add pickup and dropoff
        #     curr_plan.extend([entry_pickup, entry_dropoff])
        # else: 
        #     curr_loc = curr_plan.get_next_location() #Get current location of vehicle
        #     dist_to_new_pickup = self._geo.get_distance(curr_loc, event.location_pickup)
        #     if dist_to_new_pickup == None:
        #         print("Distance to new pickup is None, this should not happen")
        #         dist_to_new_pickup = float('inf')
        #     dist_to_new_dropoff = self._geo.get_distance(event.location_pickup, event.location_dropoff)
        #     i_pickup = 0
        #     for i in range(1,len(curr_plan._entries)):
        #         entry = curr_plan._entries[i]
        #         #If the new pickup is closer than the current entry, insert it before the current entry
        #         if self._geo.get_distance(entry.location, event.location_pickup) > dist_to_new_pickup:
        #             curr_plan._entries.insert(i, entry_pickup)
        #             i_pickup = i
        #             break
        #         if i == len(curr_plan._entries) - 1:
        #             curr_plan._entries.append(entry_pickup)
        #             i_pickup = len(curr_plan._entries) - 1
        #             break
        #     for i in range(i_pickup+1, len(curr_plan._entries)):
        #         entry = curr_plan._entries[i]
        #         #If the new dropoff is closer than the current entry, insert it before the current entry
        #         if self._geo.get_distance(entry.location, event.location_dropoff) > dist_to_new_dropoff:
        #             curr_plan._entries.insert(i+1, entry_dropoff)
        #             break
        #         if i == len(curr_plan._entries) - 1:
        #             curr_plan._entries.append(entry_dropoff)
        #             break
        #     #If there is already a passenger compute when to pickup passenger, This means the closest 
        self._vehicle_assigned_passengers[responsible_vehicle_id] += 1 #Increase number of passengers assigned to vehicle
        # print("Number of Passengers assigned to vehicle", responsible_vehicle_id, ":", self._vehicle_assigned_passengers[responsible_vehicle_id])
        if assigned_vehicle.assigned_passengers >= assigned_vehicle.max_passengers:
            # print(f"Vehicle {responsible_vehicle_id} is now full with {assigned_vehicle.assigned_passengers} passengers.")
            self._vehicle_ids_avail.remove(responsible_vehicle_id) #Remove vehicle from available vehicles, as it is now full
        return Action(vehicle_plans={responsible_vehicle_id: curr_plan}, vehicle_states={responsible_vehicle_id: VehicleState.DRIVING}) # ensure that the car is driving (it will either start or stay there)

        # # modify vehicle plan
        # pickup_location = event.location_pickup
        # vehicle_location = self._env.get_vehicle_location(self._vehicle_ids_avail[0])
        # min_distance = self._geo.get_distance(vehicle_location,pickup_location)
        # responsible_vehicle_id = self._vehicle_ids_avail[0]
        # #Assign closest free taxi
        # for free_vehicle in self._vehicle_ids_avail[1:]:
        #     vehicle_location = self._env.get_vehicle_location(free_vehicle)
        #     taxi_dist = self._geo.get_distance(vehicle_location,pickup_location)
        #     if taxi_dist < min_distance:
        #         min_distance = taxi_dist
        #         responsible_vehicle_id = free_vehicle
        # #Remove assigned taxi
        # self._vehicle_ids_avail.remove(responsible_vehicle_id)

        # # position = self._rng.integers(low=0,high=len(self._vehicle_ids))
        # # responsible_vehicle_id = self._vehicle_ids_ordered[position]

        # vehicle_plan = self._vehicle_plans[responsible_vehicle_id]
        # vehicle_plan.extend([entry_pickup, entry_dropoff])
        
        # return Action(
        #     vehicle_plans={responsible_vehicle_id: vehicle_plan},
        #     vehicle_states={responsible_vehicle_id: VehicleState.DRIVING} # ensure that the car is driving (it will either start or stay there)
        # )

    def _compute_min_distance(self,trajectory: list[(uuid.UUID,SimTime)], location: uuid.UUID) -> float:
        """
        Computes the minimum distance between a trajectory and a location.
        Parameters
        ----------
        trajectory: list[uuid.UUID)]
            List of location IDs representing the trajectory.
        location: uuid.UUID
            Location ID to compute the distance to.
        Returns
        -------
        float: Minimum distance from the trajectory to the location.
        """
        if len(trajectory) == 0:
            return None
        min_distance = float('inf')
        for loc in trajectory:
            dist = self._geo.get_distance(loc, location)
            if dist < min_distance:
                min_distance = dist
        return min_distance
    def _process_pickupdropoff(self,
                               event: PickupDropoff) -> Action:
        """
        Process PikcupDropoff Event

        Update controller car state
        Ensure that Driving (after this event the vehicle is always idle)
        """
        vehicle_plan = self._vehicle_plans[event.vehicle_id]
        assert event.location == vehicle_plan.get_next_location(), \
            f"Dude, a vehicle arrived at {event.location} but our simplest controller "\
            +f"cannot get it right and thinks the vehicle is at {vehicle_plan.get_next_location()}."
        vehicle_plan.advance_plan() #advance plan by one step

        if len(vehicle_plan.get_plan_locations())>0:
            next_state = VehicleState.DRIVING
        else:
            next_state = VehicleState.IDLING
        vehicle = self._env.get_vehicle(event.vehicle_id)
        if vehicle.assigned_passengers < vehicle.max_passengers and not self._vehicle_ids_avail.__contains__(event.vehicle_id):
            self._vehicle_ids_avail.append(event.vehicle_id)
            if len(self._backlog_requests) > 0:
                # print("Removing Request from Backlog")
                request = self._backlog_requests.pop(0)
                #Process request
                action = self._process_riderequest(request,specific_vehicle= event.vehicle_id)
                return action
        if len(self._backlog_requests) > 0 and len(self._vehicle_ids_avail):
            request = self._backlog_requests.pop(0)
            action = self._process_riderequest(request)
            action.vehicle_states[event.vehicle_id] = next_state
            return action
        return Action(
            vehicle_states={event.vehicle_id: next_state}
        )
    def update_vehicle_plans(self,
                             vehicle_plans: dict[uuid.UUID, VehiclePlan]):
        """
        Update the vehicle plan if more than one controller is acting and this controller 
        needs to keep track of vehicle plans
        """
        pass