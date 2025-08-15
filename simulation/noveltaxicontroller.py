import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np

from controller import Controller
from event import Event, RideRequest, PickupDropoff
from action import Action
from vehicle import VehiclePlan, VehiclePlanEntry, VehicleState, Vehicle
from bielegrid_geography import BieleGridGeography
from bielegrid import BieleGrid
from simetime import SimTime
from passenger import Passenger

class NovelTaxiController(Controller):
    """
    An efficient taxi controller that uses a 2-opt local search to optimize
    vehicle routes based on a detailed cost function, assigning new ride
    requests to the best available vehicle.
    """

    MAX_ASSIGNMENT_TIME = 7 * 60

    def __init__(self,
                 locations: set[uuid.UUID],
                 vehicles: set[uuid.UUID],
                 geo: BieleGridGeography,
                 env: BieleGrid,
                 max_time_factor: float = 5.0,
                 cost_wait_time: float = 1.0,
                 cost_travel_time: float = 1.0,
                 backlog_requeue_interval: float = 60):
        """
        Initializes the NovelTaxiController.

        Args:
            locations: A set of all location IDs in the simulation.
            vehicles: A set of all vehicle IDs in the simulation.
            geo: The geography of the simulation.
            env: The simulation environment.
            max_time_factor: The maximum allowed ratio of actual travel time to
                expected travel time for a passenger.
            cost_wait_time: The weight of passenger wait time in the cost
                function.
            cost_travel_time: The weight of passenger travel time in the cost
                function.
        """
        self._location_ids = locations
        self._vehicle_ids = vehicles
        self._geo = geo
        self._env = env
        self.MAX_TIME_FACTOR = max_time_factor
        self.COST_WAIT_TIME = cost_wait_time
        self.COST_TRAVEL_TIME = cost_travel_time

        self._vehicle_plans = {vid: VehiclePlan() for vid in self._vehicle_ids}
        self._available_vehicles = set(self._vehicle_ids)
        self._backlog = []
        self._current_time = SimTime(0)
        self.backlog_requeue_interval = backlog_requeue_interval #In seconds

    def process_event(self, event: Event) -> Action:
        """Processes a single event from the simulation."""
        self._current_time = SimTime(event.timepoint.time_s)

        if isinstance(event, RideRequest):
            return self._process_ride_request(event)
        elif isinstance(event, PickupDropoff):
            return self._process_pickup_dropoff(event)
        return Action()

    def set_communication_range(self,dist):
        self.max_pickup_range = dist

    def _process_ride_request(self, event: RideRequest) -> Action:
        """
        Processes a ride request by finding the best vehicle to serve it.
        """
        # if not self._available_vehicles:
        #     self._backlog.append(event)
        #     return Action()
        self._current_time = SimTime(event.timepoint.time_s)
        passenger = self._env.get_passenger(event.passenger_id)
        
        vehicles_in_range = {
            v_id for v_id in self._available_vehicles
            if self._geo.get_distance(self._env.get_vehicle(v_id).get_vehicle_location(), passenger.location_pickup) <= self.max_pickup_range
            # if self._geo.get_distance_meters(self._env.get_vehicle(v_id).get_vehicle_location(), passenger.location_pickup) <= self.max_pickup_range

        }

        if not vehicles_in_range:
            # Re-queue the request if no vehicles are in range
            new_event = RideRequest(
                timepoint=SimTime(self._current_time.time_s + self.backlog_requeue_interval),
                location_pickup=passenger.location_pickup,
                location_droppoff=passenger.location_dropoff,
                passenger_id=passenger.id
            )

            if new_event.timepoint.time_s - passenger.time_spawn.time_s > self.MAX_ASSIGNMENT_TIME:
                self._env.passenger_timeout(passenger,self._current_time)
                return Action()

            self._env._event_queue.put(new_event)
            return Action()

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._calculate_assignment_cost, v_id, passenger): v_id for v_id in vehicles_in_range}
            results = {futures[future]: future.result() for future in futures}

        best_vehicle_id, (best_cost, best_plan) = min(results.items(), key=lambda item: item[1][0])

        if best_cost == float('inf') or best_plan is None:
            new_event = RideRequest(
                timepoint=SimTime(self._current_time.time_s + self.backlog_requeue_interval),
                location_pickup=passenger.location_pickup,
                location_droppoff=passenger.location_dropoff,
                passenger_id=passenger.id
            )
            if new_event.timepoint.time_s - passenger.time_spawn.time_s > self.MAX_ASSIGNMENT_TIME:
                self._env.passenger_timeout(passenger,self._current_time)
                return Action()
            self._env._event_queue.put(new_event)
            return Action()

        self._vehicle_plans[best_vehicle_id] = best_plan
        vehicle = self._env.get_vehicle(best_vehicle_id)
        vehicle.assign_passenger(passenger.id)

        if vehicle.assigned_passengers >= vehicle.max_passengers:
            self._available_vehicles.discard(best_vehicle_id)

        return Action(vehicle_plans={best_vehicle_id: best_plan},
                      vehicle_states={best_vehicle_id: VehicleState.DRIVING})

    def _calculate_assignment_cost(self, vehicle_id: uuid.UUID, new_passenger: Passenger) -> tuple[float, VehiclePlan]:
        """
        Calculates the optimal plan and its cost for assigning a new passenger to a vehicle.
        """
        vehicle = self._env.get_vehicle(vehicle_id)
        
        passengers_onboard = [p for p in vehicle.passengers.values()]
        passengers_assigned_ids = [pid for pid in vehicle.assigned_passengers_list if pid not in vehicle.passengers]
        passengers_assigned = [self._env.get_passenger(pid) for pid in passengers_assigned_ids]
        all_passengers = passengers_onboard + passengers_assigned + [new_passenger]

        # Create a simple, valid initial plan by appending the new request.
        initial_plan = deepcopy(self._vehicle_plans[vehicle_id])
        initial_plan.append(VehiclePlanEntry(location=new_passenger.location_pickup, passenger_ids={new_passenger.id}))
        initial_plan.append(VehiclePlanEntry(location=new_passenger.location_dropoff, passenger_ids={new_passenger.id}))
        
        # Optimize the plan using 2-opt local search.
        best_cost, best_plan = self._optimize_plan_with_2opt(initial_plan, all_passengers, vehicle)

        return best_cost, best_plan

    def _optimize_plan_with_2opt(self, plan: VehiclePlan, passengers: list[Passenger], vehicle: Vehicle) -> tuple[float, VehiclePlan]:
        """
        Improves a given vehicle plan using a 2-opt local search heuristic.
        The search continues until no further improvement can be made.
        """
        best_plan = plan
        best_cost = self._compute_plan_cost(best_plan, passengers, vehicle)
        
        improved = True
        while improved:
            improved = False
            stops = best_plan.get_plan_locations()
            num_stops = len(stops)

            for i in range(num_stops - 1):
                for j in range(i + 2, num_stops):
                    # Create a new route by reversing the segment between i+1 and j
                    new_stops = stops[:i+1] + stops[i+1:j+1][::-1] + stops[j+1:]
                    
                    new_plan = self._create_plan_from_stops(new_stops, passengers)

                    if not self._is_plan_valid(new_plan, passengers, vehicle):
                        continue

                    new_cost = self._compute_plan_cost(new_plan, passengers, vehicle)

                    if new_cost < best_cost:
                        best_plan = new_plan
                        best_cost = new_cost
                        improved = True
                        break 
                if improved:
                    break
        
        return best_cost, best_plan

    def _is_plan_valid(self, plan: VehiclePlan, passengers: list[Passenger], vehicle: Vehicle) -> bool:
        """Checks if a plan is valid (pickup before dropoff for all passengers)."""
        onboard_at_start = set(vehicle.passengers.keys())
        
        for p in passengers:
            if p.id in onboard_at_start:
                continue

            pickup_idx, dropoff_idx = -1, -1
            
            for i, entry in enumerate(plan._entries):
                if p.id in entry.passenger_ids:
                    if entry.location == p.location_pickup and pickup_idx == -1:
                        pickup_idx = i
                    elif entry.location == p.location_dropoff and dropoff_idx == -1:
                        dropoff_idx = i
            
            if pickup_idx == -1 and dropoff_idx != -1: # Dropoff without pickup
                return False
            if pickup_idx != -1 and dropoff_idx == -1: # Pickup without dropoff
                return False
            if dropoff_idx < pickup_idx: # Dropoff before pickup
                return False
                
        return True

    def _create_plan_from_stops(self, stops: list[uuid.UUID], passengers: list[Passenger]) -> VehiclePlan:
        """Creates a VehiclePlan from a list of stops, assigning passengers."""
        plan = VehiclePlan()
        passenger_map = {p.id: p for p in passengers}
        
        for stop_loc in list(dict.fromkeys(stops)): # Iterate through unique stops in order
            entry = VehiclePlanEntry(location=stop_loc, passenger_ids=set())
            for p_id, p in passenger_map.items():
                if p.location_pickup == stop_loc or p.location_dropoff == stop_loc:
                    entry.passenger_ids.add(p_id)
            if entry.passenger_ids:
                plan.append(entry)
        
        return plan

    def _compute_plan_cost(self, plan: VehiclePlan, passengers: list[Passenger], vehicle: Vehicle) -> float:
        """Computes the cost of a vehicle plan from the vehicle's current state."""
        sim_time = self._current_time
        sim_loc = vehicle.get_vehicle_location()
        
        sim_passengers = {p.id: deepcopy(p) for p in passengers}
        onboard = set()
        for p_id, p in vehicle.passengers.items():
            onboard.add(p_id)
            sim_passengers[p_id].time_pickup = p.time_pickup

        total_cost = 0
        for entry in plan._entries:
            travel_duration = self._geo.predict_travel_duration(sim_loc, entry.location)
            sim_time = SimTime(sim_time.time_s + travel_duration.duration_s)
            sim_loc = entry.location

            for p_id in list(onboard):
                p = sim_passengers[p_id]
                if entry.location == p.location_dropoff and p_id in entry.passenger_ids:
                    onboard.discard(p_id)
                    travel_duration_s = sim_time.time_s - p.time_pickup.time_s
                    if travel_duration_s > self.MAX_TIME_FACTOR * p.expected_travel_time.duration_s:
                        return float('inf')
                    total_cost += self.COST_TRAVEL_TIME * (travel_duration_s - p.expected_travel_time.duration_s)
            
            for p_id in entry.passenger_ids:
                if p_id in sim_passengers and p_id not in onboard:
                    p = sim_passengers[p_id]
                    if entry.location == p.location_pickup:
                        onboard.add(p_id)
                        p.time_pickup = sim_time
                        wait_time = sim_time.time_s - p.time_spawn.time_s
                        if wait_time > self.MAX_ASSIGNMENT_TIME:
                            return float('inf')
                        total_cost += self.COST_WAIT_TIME * (sim_time.time_s - p.time_spawn.time_s)

        return total_cost

    def _process_pickup_dropoff(self, event: PickupDropoff) -> Action:
        """Processes a pickup/dropoff event and updates the vehicle's state."""
        vehicle = self._env.get_vehicle(event.vehicle_id)
        plan = self._vehicle_plans[event.vehicle_id]
        dropped_off = set(event.dropoff_passenger_ids)
        
        plan.advance_plan()
        
        # After a pickup, the plan might contain a redundant, future pickup
        # for the same passenger. We need to clean these out.
        passengers_on_board_ids = set(vehicle.passengers.keys())
        
        cleaned_entries = []
        for entry in plan._entries:
            # Check for redundant pickups and remove them.
            redundant_pickups = set()
            for pid in entry.passenger_ids:
                passenger = self._env.get_passenger(pid)
                # It's a redundant pickup if the passenger is already on board
                # and this stop is their pickup location.
                if pid in passengers_on_board_ids and entry.location == passenger.location_pickup:
                    redundant_pickups.add(pid)
                if pid in dropped_off:
                    redundant_pickups.add(pid)
            entry.passenger_ids.difference_update(redundant_pickups)

            # Only keep the entry if it still has passengers associated with it.
            if entry.passenger_ids:
                cleaned_entries.append(entry)
                
        plan._entries = cleaned_entries
        plan.squish_plan()
        
        next_state = VehicleState.DRIVING if plan.get_plan_locations() else VehicleState.IDLING
        
        if vehicle.assigned_passengers < vehicle.max_passengers:
            self._available_vehicles.add(event.vehicle_id)

        # The action must include the MODIFIED plan to update the vehicle object
        action = Action(
            vehicle_plans={event.vehicle_id: plan},
            vehicle_states={event.vehicle_id: next_state}
        )

        if self._backlog and self._available_vehicles:
            request = self._backlog.pop(0)
            backlog_action = self._process_ride_request(request)
            
            # Manually merge the actions
            if backlog_action.vehicle_plans:
                action.vehicle_plans.update(backlog_action.vehicle_plans)
            if backlog_action.vehicle_states:
                action.vehicle_states.update(backlog_action.vehicle_states)
            
        return action
    def update_vehicle_plans(self,
                             vehicle_plans: dict[uuid.UUID, VehiclePlan]):
        """
        Update the vehicle plan if more than once controller are acting
        """
        for vehicle_id,vehicle_plan in vehicle_plans.items():
           self._vehicle_plans[vehicle_id] = vehicle_plan
        pass