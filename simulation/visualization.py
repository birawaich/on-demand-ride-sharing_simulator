import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.axes import Axes
import matplotlib.animation as animation
import copy
import uuid
from collections import deque
from typing import Callable
from datetime import datetime

from event import VisualizationRequest
from simetime import SimTime, SimDuration
from eventqueue import EventQueue

class VehicleMarker:
    """
    Marker used to represent a vehicle

    Essentially a circle (based) with wedges=pie slices for as many passengers
    as this vehicle can hold.

    Provides functions to change the position as well as changing the occupancy
    """

    COLOR_BACKFILL = '#272f46ff'
    COLOR_NOTOCCUPIED = '#7b02cdff'
    COLOR_OCCUPIED = '#02cda1ff'

    RADIUS_VEHICLE = 100 #radius in m of a vehicle
    # RADIUS_VEHICLE = .001 #radius in m of a vehicle
    
    def __init__(self,
                 ax: Axes,
                 x: float,y: float,
                 occupancy: int, max_occupancy: int):
        # store stuff
        self._ax = ax

        # set parameters
        radius = VehicleMarker.RADIUS_VEHICLE
        radius_wedge = radius*0.8
        theta = 360.0/max_occupancy

        # create background circle
        self._circle = Circle((x,y),
                              radius=radius,
                              edgecolor=None,
                              facecolor=VehicleMarker.COLOR_BACKFILL)
        ax.add_patch(self._circle)
        
        # create small wedges
        self._wedges: list[Wedge] = []
        running_theta = 0.0
        for i in range(max_occupancy):
            wedge = Wedge((x,y),
                          r=radius_wedge,
                          theta1=running_theta,
                          theta2=running_theta + theta,
                          facecolor=VehicleMarker.COLOR_NOTOCCUPIED,
                          edgecolor=None)
            ax.add_patch(wedge)
            self._wedges.append(wedge)
            running_theta += theta
        self._current_occupancy: int = 0

        # set occupancy
        self._set_occupancy(occupancy)

    def _set_occupancy(self,
                       new_occupancy: int):
        """changes the colors to represent the new occupancy"""
        if new_occupancy == self._current_occupancy:
            return
        if new_occupancy < self._current_occupancy: #need to have less occupancy
            for idx in range(new_occupancy,self._current_occupancy):
                self._wedges[idx].set_facecolor(VehicleMarker.COLOR_NOTOCCUPIED)
        else: #need to have more occupancy
            for idx in range(self._current_occupancy,new_occupancy):
                self._wedges[idx].set_facecolor(VehicleMarker.COLOR_OCCUPIED)
        self._current_occupancy = new_occupancy
        return

    def update(self,
               x: float, y: float,
               occupancy: int):
        """Update the position and occupancy of a vehicle"""
        self._circle.center = (x,y)
        for wedge in self._wedges:
            wedge.set_center((x,y))
        self._set_occupancy(occupancy)

class LocationMarker:
    """
    Class to mark a location
    
    Essentially a semi-circle "holding" all waiting passengers as boxes.

    """

    PASSENGER_HEIGHT = VehicleMarker.RADIUS_VEHICLE*0.4
    """height of a passenger, in m"""

    PASSENGER_WIDTH = VehicleMarker.RADIUS_VEHICLE*2
    """width of a passenger, in m"""

    COLOR_BASE = "#03113bff"
    COLOR_PASSENGER = VehicleMarker.COLOR_OCCUPIED

    def __init__(self,
                 ax: Axes,
                 x: float,y: float):
        # store stuff
        self._ax = ax
        self._x = x
        self._y = y
        self._current_num_waiting_passengers: int = 0

        # create base
        self._base = Wedge((x,y),
            r=LocationMarker.PASSENGER_WIDTH/2,
            theta1=180,
            theta2=0,
            facecolor=LocationMarker.COLOR_BASE,
            edgecolor=LocationMarker.COLOR_BASE,
            linewidth=0.5)
        ax.add_patch(self._base)   

        # create stuff for waiting passengers
        self._stack_waitingboxes = deque()

    def update(self,
               num_waiting_passengers: int):
        """update the marker --> amount of waiting passengers"""

        if num_waiting_passengers == self._current_num_waiting_passengers:
            return
        
        delta = num_waiting_passengers - self._current_num_waiting_passengers
        if delta > 0: #add passengers
            x = self._x - LocationMarker.PASSENGER_WIDTH/2
            # peak to get the y level
            y = self._y
            if len(self._stack_waitingboxes) != 0:
                y = self._stack_waitingboxes[0].get_y()\
                            + LocationMarker.PASSENGER_HEIGHT
            # add new passenger boxes
            for i in range(0,delta):
                box = Rectangle(
                    xy=(x,y),
                    width=LocationMarker.PASSENGER_WIDTH,
                    height=LocationMarker.PASSENGER_HEIGHT,
                    angle=0.0,
                    facecolor=LocationMarker.COLOR_PASSENGER,
                    edgecolor=LocationMarker.COLOR_BASE,
                    linewidth=0.5
                )
                self._ax.add_patch(box)
                self._stack_waitingboxes.appendleft(box)
        else: #remove passengers
            for i in range(delta,0):
                box = self._stack_waitingboxes.popleft()
                box.remove()

        self._current_num_waiting_passengers = num_waiting_passengers
        return

class EnvPlot:
    """
    Class to hold plot information for the environment visualization plot

    Attributes
    ----------
    _fig
        figure
    _ax
        main axis
    _vehicles
        List of Vehilce Marker Objects
    _locations
        list of LocationMarker objects
    """

    def __init__(self,
                 func_plotbackground: Callable[[Axes], None],
                 simtime: SimTime,
                 coordinates: list[np.ndarray],
                 limits_coordinates: tuple[np.ndarray,np.ndarray],
                 vehicle_coordinates: np.ndarray,
                 vehicle_occupancies: list[int],
                 vehicle_max_occupancies: list[int]):
        """
        
        Parameters
        ----------
        coordinates: list[np.ndarray]
            List of all coordinates of locations
        limits_coordinates: tuple[np.ndarray,np.ndarray]
            Tuple of minimal and maximal coordinates of locations
        func_plotbackground: Callable[[Axes], None]
            Function that given some axes plots a map background.
        """
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title("Environment Visualization")
        self._fig = fig
        self._ax = ax
        # ax.grid(True)

        num_vehicles = np.size(vehicle_coordinates,1)

        #plot background
        func_plotbackground(ax)

        #plot locations
        self._locations: list[LocationMarker] = []
        for coordinate in coordinates:
            self._locations.append(LocationMarker(
                ax=ax,
                x=coordinate[0],
                y=coordinate[1]
            ))

        #plot vehicles
        self._vehicles: list[VehicleMarker] = []
        for idx_vehicle in range(num_vehicles):
            self._vehicles.append(VehicleMarker(
                ax=ax,
                x=vehicle_coordinates[0,idx_vehicle],
                y=vehicle_coordinates[1,idx_vehicle],
                occupancy=vehicle_occupancies[idx_vehicle],
                max_occupancy=vehicle_max_occupancies[idx_vehicle]
            ))
        
        coords_min, coords_max = limits_coordinates
        delta = VehicleMarker.RADIUS_VEHICLE*5 #amount of m that is added for better visibility
        ax.set_ylim(coords_min[1]-delta,coords_max[1]+delta)
        ax.set_xlim(coords_min[0]-delta,coords_max[0]+delta)
        ax.set_aspect('equal')

        # Cosmetics
        fig.suptitle("Environment Visualization")
        ax.set_title(f"Map at {simtime}",fontsize=12)
        ax.set_xlabel("Longitude in m")
        ax.set_ylabel("Latitude in m")

        if EnvironmentVisualization.DO_LIVE_VISUALIZATION:
            # ensure that shows for the live visualization
            plt.show(block=False)
            plt.pause(0.01) #just ensures that window shows up TODO do better?

    def update_plot(self,
                    simtime: SimTime,
                    vehicle_coodrinates: np.ndarray,
                    vehicle_occupancies: list[int],
                    waiting_passenger: list[int],
                    in_live_visualization: bool = True):
        """
        Update the plot with new data

        Parameters
        ----------
        vehicle_coodrinates: np.ndarray
            coordinate of the vehicles (first dimension: x,y; 2nd dimension: vehicle)
        simtime: SimTime
            time of this update
        vehicle_occupancies: list[int])
            occupancy in each vehicle (same order as above)
        waiting_passengers: list[int]
            amount of waiting passengers at each location (same order as static values)
        in_live_visualization: bool
            set to true if in a live visualization, false otherwise
        """
        # update vehicles
        self._ax.set_title(f"Map at {simtime}",fontsize=12)
        for idx, vehicle in enumerate(self._vehicles):
            vehicle.update(
                x=vehicle_coodrinates[0,idx],
                y=vehicle_coodrinates[1,idx],
                occupancy=vehicle_occupancies[idx]
            )

        # update locations
        for idx, location in enumerate(self._locations):
            location.update(
                num_waiting_passengers=waiting_passenger[idx]
            )

        if in_live_visualization:
            plt.pause(0.001)

class  EnvironmentVisualization:
    """
    Class to handle environment visualization

    Attributes
    ----------
    visualization_interval_s: flaot
        Time resolution of the visualization in seconds
        Note: cannot be changed during execution
    _timepoint_last_visualization_request: SimTime
        (private) timepoint of the last visualization
    _plot_index: int
        (private) what index from the storage is currently shown
    _vehicle_max_occpancies: dict[uuid.UUID, int]
        (private) dictionary that mapes vehicle_id --> maximal amout of passenter in a vehicle
    _location_coordinates: dict[uuid.UUID, np.ndarray]
        (private) dictinary of all location coordiantes

    _plot_times: list[SimTime]
        (private) what index corresponds with what time
    _vehicle_positions: dict[uuid.UUID, list[np.array]]
        (private) dictionary storying all positions of the vehicles over time
    _vehicle_occupancies: dict[uuid.UUID, list[int]]
        (private) Dictionary that maps vehicle_id --> list of #passengers in vehicle
    _waiting_passengers: dict[uuid.UUID, int]
        (private) Dictionary that maps location_id --> amount of passengers waiting there

    _plot: EnvPlot
        (private) custom object doing all the dirty plot work
    _func_plotbackground: Callable[[Axes], None]
        (private) function handle to plot a map
    _limits_coordinates: tuple[np.ndarray,np.ndarray]
        (private) Tuple of minimal and maximal coordinates of locations (min,max)

    """

    ### CLASS PROPERTIES ###

    DO_LIVE_VISUALIZATION = False
    """Set to True to do a live visualization. Values will be stored regardless."""

    DEBUG_PRINT = False

    ### FUNCTIONS ###

    def __init__(self,
                 queue: EventQueue):
        self.visualization_interval: SimDuration = SimDuration(5.0)

        # set status
        self._timepoint_last_visualization_request: SimTime = SimTime(0.0) #by convention
        self._plot_index: int = 0

        # set static knowledge
        self._vehicle_max_occpancies = dict()
        self._location_coordinates = dict()

        # set storage
        self._plot_times = []
        self._vehicle_positions = dict()
        self._vehicle_occupancies = dict()
        self._waiting_passengers = dict()

        self._plot = None #plot object

        # preload queue
        queue.put(VisualizationRequest(timepoint=SimTime(0)))

    def finalize_init(self,
                      coordinates: dict[uuid.UUID, np.ndarray],
                      limits_coordinates: tuple[np.ndarray,np.ndarray],
                      vehicle_positions: dict[uuid.UUID, np.array],
                      vehicle_occupancies: dict[uuid.UUID, int],
                      vehicle_max_occupancies: dict[uuid.UUID, int],
                      waiting_passengers:  dict[uuid.UUID,int],
                      func_plotbackground: Callable[[Axes], None]
                      ):
        """
        Finalize the initialization, to be called from the specific environment

        Notably, does
        - set up storage
        - does first plot (note: can change this later to allow for offline visualization)

        Parameters
        ----------
        coordinates: dict[uuid.UUID, np.ndarray]
            dictionary of location_id --> coordinate of this location
        limits_coordinates: tuple[np.ndarray,np.ndarray]
            Tuple of minimal and maximal coordinates of locations (min,max)
        vehicle_positions: dict[uuid.UUID, np.array]
            Dictionary that maps vehicle_id --> vehicle position
        vehicle_occupancies: dict[uuid.UUID, [int]]
            Dictionary that maps vehicle_id --> #passengers in vehicle
        vehicle_max_occupancies: dict[uuid.UUID, int]
            Dictionary that maps vehicle_id --> #maximal passengers in vehicle
            Note: only sent once as this is assumed to not change during the simulation
        waiting_passengers: dict[uuid.UUID,[int]]
            dictionary mapping location --> amount of waiting passengers
        func_plotbackground: Callable[[Axes], None]
            Function that given some axes plots a map background.
        """
        # load initial vehicle positions
        vehicle_positions = {vehicle_id: [position]
                            for vehicle_id, position in vehicle_positions.items()}
        
        # add keys to storage dictionaries and store static knowledge
        # Note: this ensures that all the storage dictionaries have the same ordering of IDs!
        vehicle_occupancies = []
        for vehicle_id in vehicle_positions:
            self._vehicle_positions[vehicle_id] = []
            self._vehicle_occupancies[vehicle_id] = []
            self._vehicle_max_occpancies[vehicle_id] = vehicle_max_occupancies[vehicle_id]
            vehicle_occupancies.append(vehicle_max_occupancies[vehicle_id])

        self._location_coordinates = copy.copy(coordinates)
        for location_id in self._location_coordinates: #this ensures the same ordering
            self._waiting_passengers[location_id] = []

        # store stuff
        self._func_plotbackground: Callable[[Axes], None] = func_plotbackground
        self._limits_coordiantes: tuple[np.ndarray,np.ndarray] = limits_coordinates
        
        # create plot
        if EnvironmentVisualization.DO_LIVE_VISUALIZATION:
            initial_vehicle_coordinates = self._transform_vehicle_coordinates(vehicle_positions=vehicle_positions)
            location_coordinates = [
                coords for coords in self._location_coordinates.values()
            ]

            self._plot: EnvPlot = EnvPlot(
                simtime=SimTime(0),
                func_plotbackground=func_plotbackground,
                coordinates=location_coordinates,
                limits_coordinates=limits_coordinates,
                vehicle_coordinates=initial_vehicle_coordinates,
                vehicle_max_occupancies=list(self._vehicle_max_occpancies.values()),
                vehicle_occupancies=vehicle_occupancies)
            # do not update index as nothing got added to storage

    def update(self,
               sim_time: SimTime,
               vehicle_positions: dict[uuid.UUID, np.array],
               vehicle_occupancies: dict[uuid.UUID, int],
               waiting_passengers: dict[uuid.UUID, int]):
        """
        Udpate the Visualization

        Adds the data to the storage, and (if doing live visualization) update
        the live plot

        Parameters
        ----------
        sim_time: SimTime
            time during which this is called
        vehicle_positions: dict[uuid.UUID, np.array]
            Dictionary that maps vehicle_id --> vehicle position
        vehicle_occupancies: dict[uuid.UUID, int]
            Dictionary that maps vehicle_id --> #passengers in vehicle
        waiting_passengers:  dict[uuid.UUID,int]
            dictionary mapping location --> amount of waiting passengers
            Does not need to include all locations!
        """
        if EnvironmentVisualization.DEBUG_PRINT:
            EnvironmentVisualization._debug_print(f"Update visualization @ {sim_time}, storing to index {self._plot_index}")

        # add time
        assert len(self._plot_times) == self._plot_index, \
            f"There should be {self._plot_index} timepoints already, but there is {len(self._plot_times)}!"
        self._plot_times.append(copy.copy(sim_time))

        # add vehicle positions
        for vehicle_id, position in vehicle_positions.items():
            self._vehicle_positions[vehicle_id].append(position)
        # add vehicle occupancies
        for vehicle_id, occupancy in vehicle_occupancies.items():
            self._vehicle_occupancies[vehicle_id].append(occupancy)
        # add amount of waiting passengers
        for location_id in self._location_coordinates:
            #first add 0 waiting passengers everywhere
            self._waiting_passengers[location_id].append(int(0))
        for location_id, num_waiting in waiting_passengers.items():
            # then update the ones that are present
            self._waiting_passengers[location_id][self._plot_index] = num_waiting

        # trigger update of visualization
        if EnvironmentVisualization.DO_LIVE_VISUALIZATION:
            self._plot.update_plot(simtime=sim_time,
                                vehicle_coodrinates=self._transform_vehicle_coordinates(),
                                vehicle_occupancies=self._transform_vehicle_occpancy(),
                                waiting_passenger=self._transform_waiting_passengers())

        # increase the index AFTER plotting
        self._plot_index += 1

    def create_animation(self) -> animation.FuncAnimation:
        """
        Createas an animation from the values in the storage
        """
        if EnvironmentVisualization.DEBUG_PRINT:
            EnvironmentVisualization._debug_print("Creating Animation...")

        # create figure
        if self._plot is None:
            # prepare values
            location_coordinates = [
                coords for coords in self._location_coordinates.values()
            ]
            true_index = self._plot_index #quickfix, adjust the index to plot the inital values
            self._plot_index = 0
            #create initial plot
            self._plot: EnvPlot = EnvPlot(
                simtime=SimTime(0),
                func_plotbackground=self._func_plotbackground,
                coordinates=location_coordinates,
                limits_coordinates=self._limits_coordiantes,
                vehicle_coordinates=self._transform_vehicle_coordinates(),
                vehicle_max_occupancies=list(self._vehicle_max_occpancies.values()),
                vehicle_occupancies=self._transform_vehicle_occpancy())
            self._plot_index = true_index

        fig = self._plot._fig

        # assert that the right amount of things are stored
        num_frames = len(self._plot_times)
        for id, positions in self._vehicle_positions.items():
            assert len(positions) == num_frames, \
                f"Aim to have {num_frames} but got {len(positions)} amount of positions for vehicle {id}"
        for id, occupancies in self._vehicle_occupancies.items():
            assert len(occupancies) == num_frames, \
                f"Aim to have {num_frames} but got {len(occupancies)} amount of occupancies for vehicle {id}"
        for id, waiting_passengers in self._waiting_passengers.items():
            assert len(waiting_passengers) == num_frames, \
                f"Aim to have {num_frames} but got {len(waiting_passengers)} amount of waiting_passengers for location {id}"

        # create animation
        fps = 30
        ani = animation.FuncAnimation(
            fig=fig,
            func=self._animation_update,
            frames=num_frames,
            blit=False,
            interval=int(1000/fps)
        )
        # plt.show(block=False)

        # store
        if EnvironmentVisualization.DEBUG_PRINT:
            EnvironmentVisualization._debug_print("Exporting Animation...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"out/environment-animation_{timestamp}.mp4"
        ani.save(filename=filename,
                 writer='ffmpeg',
                 fps=fps,
                 dpi=300) #default is 100, inches if from the figsize
        print(f"Exported Environment Visualization to {filename}.")

    def potentially_add_visualization_request(self,
                                              queue: EventQueue):
        """
        Depending on internatal status, add a visualization request to the EventQueue

        Note: does not care when last insertion was, just adds a new one. Only call once 
              a visualization request was popped from the queue.

        Parameters
        ----------
        queue: EventQueue
            EventQueue where events should be added, from environment
        """                                              
        
        # do not add anything if the queue only contains visualization requests
        if queue.no_action_events_in_queue():
            return
        
        # add visualization request
        queue.put(VisualizationRequest(timepoint=
                                       copy.copy(self._timepoint_last_visualization_request
                                       .add_duration(self.visualization_interval))))
        #note: bit of python magic here: the timepoint is automatically updated, 
        #      and then copied to be its own timepoint in the event (yes, this is my own madness, not some LLM)
        return
    
    ### PRIVATE

    def _transform_vehicle_coordinates(self,
                                       vehicle_positions = None) -> np.ndarray:
        """
        Transforms the vehicle coordinates of the current plotting index s.t. it can be plotted
        Or for the supplied vehicle positions
        """
        if vehicle_positions is None:
            vehicle_positions = self._vehicle_positions

        coordinates = [coord_list[self._plot_index] for coord_list in vehicle_positions.values()]
        result = np.empty((2,len(coordinates)))
        for i, coordinate in enumerate(coordinates):
            result[:,i] = coordinate
        return result
    
    def _transform_vehicle_occpancy(self) -> list[int]:
        """
        Transfors the vehicle occupancy of the current plotting index s.t. it can be plotted
        """
        return [occupancy_list[self._plot_index] for occupancy_list in self._vehicle_occupancies.values()]
    
    def _transform_waiting_passengers(self) -> list[int]:
        """
        Transforms the waiting passengers of the current plotting index s.t. it can be plotted
        """
        return [
            num_waiting_passengers[self._plot_index]
            for num_waiting_passengers in self._waiting_passengers.values()
        ]
    
    def _animation_update(self,frame: int):
        self._plot_index = frame
        self._plot.update_plot(
            simtime=self._plot_times[frame],
            vehicle_coodrinates=self._transform_vehicle_coordinates(),
            vehicle_occupancies=self._transform_vehicle_occpancy(),
            waiting_passenger=self._transform_waiting_passengers(),
            in_live_visualization=False
        )

    ### DEBUG ###
    def _debug_print(str: str):
        print("[EnvVisual] "+str)




