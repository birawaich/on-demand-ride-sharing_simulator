import uuid
import numpy as np
import pandas as pd
from enum import Enum, auto
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import seaborn as sns
import pickle as pkl
import time
# import scienceplots
from datetime import datetime

from passenger import Passenger
from vehicle import Vehicle, VehicleState
from geo import Geography
from simetime import SimTime, SimDuration


class DataIdentifier(Enum):
    """Enumerated Type to describe different data"""

    PASSENGER_RECORDED_TRIP_DURATION = auto()
    """passenger records of how long each trip lasted"""

    PASSENGER_MINIMAL_TRIP_DURATION = auto()
    """passenger records of the minimal time of each trip"""

    PASSENGER_RECORDED_WAITING_TIME = auto()
    """passenger records of how long each passenger had to wait"""

    PASSENGER_TRIP_OPTIMALITY_NO_WAITING_TIME = auto()
    """passenger records of how optimal the trip duration was (without waiting times), a factor >= 1"""

    PASSENGER_TRIP_OPTIMALITY_WITH_WAITING_TIME = auto()
    """passenger records of how optimal the trip duration was with waiting times, a factor >= 1"""

    VEHICLE_NUM_VEHICLE_DRIVING_AT_TIME = auto()
    """vehicle records of how many vehciles are driving at a certain time"""

    VEHICLE_NUM_VEHICLE_IDLING_AT_TIME = auto()
    """vehicle records of how many vehciles are idling at a certain time"""

    VEHICLE_NUM_PASSENGER_RIDING_AT_TIME_TOTAL = auto()
    """vehicle records of how many passengers are riding in all vehicles at a certain time"""

    VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_TOTAL = auto()
    """vehicle records of how many passengers are in all vehicles at a certain time (regardless what vehicle is doing)"""
    
    VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_PER_VEHICLE = auto()
    """vehicle records of how many passengers are in all vehicles at a certain time 
    (regardless what vehicle is doing) per vehicle
    
    Format: [vehicle, Timepoint] -> nuimber of passengers there"""

    VEHCILE_TIME_IN_MODE_PER_VEHICLE = auto()
    """Total summed time each vehicle is a certain mode. 
    A mode goes from 0 to 2*#Pm+1 where #Pm denotes the maximum amount of passengers of a vehicle
    mode 0-#Pm ... vehicle is idling with mode passengers on board
    mode #PM+1-#PM+2 ... vehilce is driving with mode-(#PM+2) passenmgers on board
    
    Format: [Vehicle, Mode] -> time in there"""

    LOCATION_WAITING_PASSENGERS_AT_TIME_TOTAL = auto()
    """amount of passengers waiting at locations at a certain time, summed over all locations"""

    CALCULATION_CONTROLLER_TIME_AT_SIM_TIME = auto()
    """how long the controller took to compute something at the sim time, all in s
    
    Format (simulation time, computation time)"""

    CALCULATION_CONTROLLER_TIME_AT_REAL_TIME = auto()
    """how long the controller took to compute something at the real time, all in s
    
    Format (real time, computation time)"""

    CALCULATION_CONTROLLER_REALTIME_VS_SIMTIME = auto()
    """at a certain controller computation: what was the real time vs. the sim time, all in s
    
    Format (real time, sim time)"""

    CALCULATION_CONTROLLER_TIMES = auto()
    """controller calculation times in s"""

    CALCULATION_ENVIRONMENT_TIME_AT_SIM_TIME = auto()
    """how long the environment took to compute something at the sim time, all in s
    
    Format (simulation time, computation time)"""

    CALCULATION_ENVIRONMENT_TIME_AT_REAL_TIME = auto()
    """how long the environment took to compute something at the real time, all in s
    
    Format (real time, computation time)"""

    CALCULATION_ENVIRONMENT_REALTIME_VS_SIMTIME = auto()
    """at a certain environment computation: what was the real time vs. the sim time, all in s
    
    Format (real time, sim time)"""

    CALCULATION_ENVIRONMENT_TIMES = auto()
    """environment calculation times in s"""

class FormatHelper:
    """Class to help formating times"""
    def __init__(self):
        raise NotImplementedError("This is a static utility class and should not be instantiated.")
    @staticmethod
    def format_duration_min(x, _=None):
        """format time duration in s into minutes
        can also be used to just format any duration"""
        return f"{x/60:.2f}"
    @staticmethod
    def format_duration_h(x, _=None):
        """format time duration in s into hours
        can also be used to just format any duration"""
        return f"{x/(60*60):.2f}"
    @staticmethod
    def format_duration_min_s(x, _=None):
        """format time duration in s into minutes and seconds
        can also be used to just format any duration"""
        minutes = int(x) // 60
        seconds = int(x) % 60
        return f'{minutes}min {seconds:02d}s'
    @staticmethod
    def format_time(x, _=None):
        """format time point in s into a clock format from the input
        can also be used to just format any duration"""
        min_total = x/60
        h_total = min_total/60
        days = int(h_total//24) +1 #amount of days, start with day 1 because calendars where not made by computer scientists
        h = int(h_total % 24)
        m = int(min_total % 60)
        return f'{days} {h:02d}:{m:02d}'
    @staticmethod
    def format_time_precise(x, _=None):
        """format time point in s into a clock format from the input down to the second (without days)
        can also be used to just format any duration"""
        min_total = x/60
        h_total = min_total/60
        h = int(h_total % 24)
        m = int(min_total % 60)
        s = int(x % 60)
        return f'{h:d}:{m:02d}:{s:02d}'
    @staticmethod
    def format_time_duration(x, _=None):
        """format time duration in s into a clock format from the input
        can also be used to just format any duration"""
        min_total = x/60
        h_total = min_total/60
        days = int(h_total//24)
        h = int(h_total % 24)
        m = int(min_total % 60)
        return f'{days}d {h:02d}h {m:02d}min'

class PassengerRecord:
    """
    Record of a single passenger

    Attributes
    ----------
    record: np.ndarray
        Records as a numpy array
        0 ... waiting time in s
        1 ... travel time in s
        2 ... minimal journey time in s
        3 ... journey efficiency without waiting times
        4 ....journey efficiency with waiting times
    """

    NUM_ENTRIES = 5
    """amount of entries per record"""

    def __init__(self,
                 passenger: Passenger,
                 minimal_journey_time_s: float = np.nan):
        record = np.empty(PassengerRecord.NUM_ENTRIES)

        # write values over to records
        record[0] = passenger.time_pickup.time_s - passenger.time_spawn.time_s #waiting time
        record[1] = passenger.time_dropoff.time_s - passenger.time_pickup.time_s #journey time
        record[2] = minimal_journey_time_s
        record[3] = record[1]/record[2] #efficiency without wiaint time
        record[4] = (record[0]+record[1])/record[2] #efficeincy with waiting time
        self.record = record

class PassengerMetrics:
    """
    Manage Metrics = Measures over Passengers

    Attributes
    ----------
    _passenger_records: list[PassengerRecord]
        list of passenger records
    _passenger_ids_with_records: set[uuid.UUID]
        set of which passengers are already in the records
    _data_matrix_cache: np.ndarray
        Data Matrix for quick access (built on demand)
        "wide, fat matrix" --> each record is a column (as numpy is row-major)
    """
    
    def __init__(self):
        self._passenger_records: list[PassengerRecord] = []
        self._passenger_ids_with_records: set[uuid.UUID] = set()
        self._data_matrix_cache: np.ndarray = None

    def __str__(self):
        # get data
        data_trip = self.get_passenger_data(DataIdentifier.PASSENGER_RECORDED_TRIP_DURATION)
        data_waiting = self.get_passenger_data(DataIdentifier.PASSENGER_RECORDED_WAITING_TIME)
        num_passengers = len(data_trip)
        mean_trip = np.mean(data_trip)
        max_trip = np.max(data_trip)
        mean_waiting = np.mean(data_waiting)
        max_waiting = np.max(data_waiting)
        # formulate string
        return f"Passenger Data:\n"+\
            f"\tNumber of Passengers: {num_passengers}\n"+\
            f"\tRide Time:\n"+\
            f"\t\tmean: {FormatHelper.format_duration_min_s(mean_trip)}\n"+\
            f"\t\tmax:  {FormatHelper.format_duration_min_s(max_trip)}\n"+\
            f"\tWaiting Time:\n"+\
            f"\t\tmean: {FormatHelper.format_duration_min_s(mean_waiting)}\n"+\
            f"\t\tmax:  {FormatHelper.format_duration_min_s(max_waiting)}\n"
    def __repr__(self):
        return str(self)

    def add_records(self,
                    record: PassengerRecord,
                    passenger_id: uuid.UUID):
        assert passenger_id not in self._passenger_ids_with_records, \
            f"Attempting to add records of passenger {passenger_id} to passenger records even though already present!"
        
        self._passenger_records.append(record)
        
        self._passenger_ids_with_records.add(passenger_id)

    def _get_data_matrix(self) -> np.ndarray:
        """Returns the data matrix by building it in case there are more records available"""
        if self._data_matrix_cache is not None \
            and self._data_matrix_cache.shape[1] == len(self._passenger_ids_with_records):
            return self._data_matrix_cache
        
        #rebuild cache
        res = np.empty((PassengerRecord.NUM_ENTRIES,len(self._passenger_ids_with_records)))
        for i, record in enumerate(self._passenger_records):
            res[:,i] = record.record
        self._data_matrix_cache = res

        return res 

    def get_passenger_data(self,
                           identifier: DataIdentifier) -> np.ndarray:
        """
        Return the measured journey duration
        """
        index: int = 0
        if identifier == DataIdentifier.PASSENGER_MINIMAL_TRIP_DURATION:
            index = 2
        elif identifier == DataIdentifier.PASSENGER_RECORDED_TRIP_DURATION:
            index = 1
        elif identifier == DataIdentifier.PASSENGER_RECORDED_WAITING_TIME:
            index = 0
        elif identifier == DataIdentifier.PASSENGER_TRIP_OPTIMALITY_NO_WAITING_TIME:
            index = 3
        elif identifier == DataIdentifier.PASSENGER_TRIP_OPTIMALITY_WITH_WAITING_TIME:
            index = 4
        else:
            raise NotImplementedError(f"Did not implement getting data from the passenger metrics with the identifier {identifier}")

        data_matrix = self._get_data_matrix()
        return data_matrix[index,:]

class VehicleRecord:
    """
    Store a vehicle record (corresponding to a specific time) for either state or num passenger

    Attributes
    ----------
    record: np.ndarray
        records for each vehicle
    """
    def __init__(self,
                 num_vehicles: int = None,
                 value: int = None,
                 index: int = None,
                 last_record: 'VehicleRecord' = None):
        """
        either: create a new record based on the last record and update the value and index

        or: creae a new record and fill it with the value
        """
        # build record
        if last_record is None:
            assert num_vehicles is not None and value is not None, \
                "need to know how large the record is somehow and what to put in it"
            record = np.full(num_vehicles,
                             value,
                             dtype=np.int_)
        else:
            record = last_record.record.copy()

        # update value
        if value is not None and index is not None:
            record[index] = value

        #store record
        self.record = record

    def adjust(self,
               value: int,
               index: int):
        """adjust the value at index"""
        self.record[index] = value

class VehicleMetrics:
    """
    Manages Vehicle metrics

    Attributes
    ----------
    _timepoints: list[float]
        points where there is recordings, time in s from start
    _vehicle_id_index: dict[uuid.UUID, int]
        index mapping the vehicle id --> index in storage
    _vehicle_state_records: list[VehicleRecord]
        vehicle state records
    _vehicle_nump_records: list[VehicleRecord]
        vehicle passenger number records

    _waiting_passengers: list[int]
        amount of passengers waiting in total
    _timepoints_waiting_passengers: list[float]
        times corresponding to amount of passengers waiting

    _data_matrix_vehicle_state_cache: np.ndarray
        data matrix to store the vehicle state records, access via function to rebuild on demand
        row: vehicle (as in index), column: time
    _data_matrix_vehicle_state_cache: np.ndarray
        data matrix to store the vehicle number of passengers records, access via function to rebuild on demand
        row: vehicle (as in index), column: time
    """

    def __init__(self,
                 vehicle_ids: set[uuid.UUID]):
        "stores an initial value at the convention start (idle, no passenger)"
        self._timepoints: list[float] = [0.0]
        
        # generate index
        self._vehicle_id_index = {
            id: i
            for i, id in enumerate(list(vehicle_ids))
        }

        self._vehicle_state_records: list[VehicleRecord] = [
            VehicleRecord(num_vehicles=len(self._vehicle_id_index),
                          value=int(VehicleState.IDLING))
        ]
        self._vehicle_nump_records: list[VehicleRecord] = [
            VehicleRecord(num_vehicles=len(self._vehicle_id_index),
                          value=int(0))
        ]
        self._waiting_passengers: list[int] = [0]
        self._timepoints_waiting_passengers: list[float] = [0.0]

        # set up empty data matrices
        self._data_matrix_vehicle_state_cache: np.ndarray = None
        self._data_matrix_vehicle_nump_cache: np.ndarray = None

    def __str__(self):
        # get data
        data_modes = self.get_vehicle_data(DataIdentifier.VEHCILE_TIME_IN_MODE_PER_VEHICLE)
        data_modes_avg = np.mean(data_modes,0)
        time_idle = data_modes_avg[0]
        time_active = sum(data_modes_avg) - time_idle
        total_time = np.sum(data_modes_avg)
        data_mode_raw = self._get_data_matrix_vehicle_state()*self._get_data_matrix_vehicle_state()
        terminated = np.all(data_mode_raw[:,-1] == 0)
        # formulate string
        return f"Vehicle Data:\n"+\
            f"\tTerminal State is all Idling: {'YES' if terminated else 'NO'}\n"+\
            f"\tActivity Share:\n"+\
            f"\t\tmean riding time: {100*time_active/total_time:.1f}% ({FormatHelper.format_duration_min_s(time_active)})\n"+\
            f"\t\tmean idling time: {100*time_idle/total_time:.1f}% ({FormatHelper.format_duration_min_s(time_idle)})\n"
    def __repr__(self):
        return str(self)

    def add_record(self,
                   time_s: float,
                   state_old: VehicleState,
                   state_new: VehicleState,
                   num_p_old: int,
                   num_p_new: int,
                   vehicle_id: uuid.UUID):
        """
        Add a record of either a state or a passenger difference to the records

        Note: only changes the variables where there is actually a difference,
              all other values are just copied from the last timestep
        """

        is_state = state_new != state_old
        is_num_p = num_p_new != num_p_old

        if not is_state and not is_num_p: #no change
            return
        
        # ensure that time is monotone and see whether only need to add
        assert self._timepoints[-1] <= time_s, \
            f"Time is monotone! Wanted to add something at timepoint {time_s}s but already have something at {self._timepoints[-1]}s"
        # find right ID
        index_vehicle = self._vehicle_id_index[vehicle_id]

        # assert that old value is actually in there
        if is_state:
            assert  self._vehicle_state_records[-1].record[index_vehicle] == state_old, \
                f"Oh no, old state should be {state_old} according to call but stored {self._vehicle_state_records[-1].record[index_vehicle]}..."
        if is_num_p:
            assert  self._vehicle_nump_records[-1].record[index_vehicle] == num_p_old, \
                f"Oh no, old numper of passenger should be {num_p_old} according to call but stored {self._vehicle_nump_records[-1].record[index_vehicle]}..."

        # if time point is already present, just add
        if  self._timepoints[-1] == time_s:
            if is_state:
                self._vehicle_state_records[-1].adjust(value=int(state_new),index=index_vehicle)
            if is_num_p:
                self._vehicle_nump_records[-1].adjust(value=num_p_new,index=index_vehicle)
            return
        
        # build new record
        if is_state:
            record_state = VehicleRecord(
                value=int(state_new),
                index=index_vehicle,
                last_record=self._vehicle_state_records[-1]
            )
        else:
            record_state = VehicleRecord(last_record=self._vehicle_state_records[-1])

        if is_num_p:
            record_passenger = VehicleRecord(
                value=num_p_new,
                index=index_vehicle,
                last_record=self._vehicle_nump_records[-1]
            )
        else:
            record_passenger = VehicleRecord(last_record=self._vehicle_nump_records[-1])

        # add new records
        self._vehicle_state_records.append(record_state)
        self._vehicle_nump_records.append(record_passenger)
        self._timepoints.append(time_s)

    def add_record_waiting(self,
                           time_s: float,
                           delta: int):
        """
        add a record of how much the total amount of passengers waiting
        at time_s changed

        Note:
        -   decided to not store this for each location as this is hardly needed
            if it is needed, it can be implemented with a VehicleRecord as well!
        -   this is a seperate function from `add_record()` as the time steps do 
            not necessarily need to be the same as
            multiplication with vehicle states or amount of ppeople in vehicle 
            simpy does not make sense
        """
         # ensure that time is monotone
        assert self._timepoints_waiting_passengers[-1] <= time_s, \
            f"Time is monotone! Wanted to add something at timepoint {time_s}s but already have something at {self._timepoints_waiting_passengers[-1]}s"
        
        # see if only need to update
        if self._timepoints_waiting_passengers[-1] == time_s:
            self._waiting_passengers[-1] += delta
            return
        
        self._waiting_passengers.append(self._waiting_passengers[-1]+delta)
        self._timepoints_waiting_passengers.append(time_s)

        
    def _get_data_matrix_vehicle_state(self) -> np.ndarray:
        """Returns the data matrix for vehicle states by building it in case there are more records available"""
        if self._data_matrix_vehicle_state_cache is not None \
            and self._data_matrix_vehicle_state_cache.shape[1] == len(self._timepoints):
            return self._data_matrix_vehicle_state_cache
        
        #rebuild cache
        res = np.empty((len(self._vehicle_id_index),len(self._timepoints)))
        for i, record in enumerate(self._vehicle_state_records):
            res[:,i] = record.record
        self._data_matrix_vehicle_state_cache = res
        return res
    
    def _get_data_matrix_vehicle_nump(self) -> np.ndarray:
        """Returns the data matrix for vehicle number of passengers by building it in case there are more records available"""
        if self._data_matrix_vehicle_nump_cache is not None \
            and self._data_matrix_vehicle_nump_cache.shape[1] == len(self._timepoints):
            return self._data_matrix_vehicle_nump_cache
        
        #rebuild cache
        res = np.empty((len(self._vehicle_id_index),len(self._timepoints)))
        for i, record in enumerate(self._vehicle_nump_records):
            res[:,i] = record.record
        self._data_matrix_vehicle_nump_cache = res
        return res
    
    def get_vehicle_time_data(self,
                              identifier: DataIdentifier) -> tuple[np.ndarray,np.ndarray]:
        """
        Returns the specified data pair: Time Values (in s) and specified data
        """
        time = np.array(self._timepoints)
        data = None
        num_data_points = -1

        # check assumptions about state that are exploited here
        assert int(VehicleState.DRIVING) == 1, "Oh no, this is needed here!"
        assert int(VehicleState.IDLING) == 0, "Oh no, this is needed here!"

        if identifier == DataIdentifier.VEHICLE_NUM_VEHICLE_DRIVING_AT_TIME:
            data = np.sum(self._get_data_matrix_vehicle_state(),0)
            num_data_points = len(data)
        elif identifier == DataIdentifier.VEHICLE_NUM_VEHICLE_IDLING_AT_TIME:
            data = len(self._vehicle_id_index) \
                    -  np.sum(self._get_data_matrix_vehicle_state(),0)
            num_data_points = len(data)
        elif identifier == DataIdentifier.VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_TOTAL:
            data = np.sum(self._get_data_matrix_vehicle_nump(),0)
            num_data_points = len(data)
        elif identifier == DataIdentifier.VEHICLE_NUM_PASSENGER_RIDING_AT_TIME_TOTAL:
            nump = self._get_data_matrix_vehicle_nump()
            state = self._get_data_matrix_vehicle_state()
            data = np.sum(nump*state,0) #this is elementwise and works due to the assumptions
            num_data_points = len(data)
        elif identifier == DataIdentifier.VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_PER_VEHICLE:
            data = self._get_data_matrix_vehicle_nump()
            num_data_points = data.shape[1]
        elif identifier == DataIdentifier.LOCATION_WAITING_PASSENGERS_AT_TIME_TOTAL:
            data = np.array(self._waiting_passengers)
            num_data_points = len(data)
            time = np.array(self._timepoints_waiting_passengers)
        else:
            raise NotImplementedError(f"Did not implement getting data from the vehcile metrics with time data with the identifier {identifier}")
        
        #assert that the sizes are right
        assert len(time) == num_data_points, \
            f"Houston, we got data of different length: {len(time)} timepoints vs. {num_data_points} datapoints (also check if the number of data points was calculated correctly...)."

        return time, data
    
    def get_vehicle_data(self,
                         identifier: DataIdentifier) -> np.ndarray:
        """
        Returns the specified data (data not related to time)
        """

        # check assumptions about state that are exploited here
        assert int(VehicleState.DRIVING) == 1, "Oh no, this is needed here!"
        assert int(VehicleState.IDLING) == 0, "Oh no, this is needed here!"

        if identifier == DataIdentifier.VEHCILE_TIME_IN_MODE_PER_VEHICLE:
            # prepare values
            nump = self._get_data_matrix_vehicle_nump()
            state = self._get_data_matrix_vehicle_state()
            state_boolean = np.astype(state,bool)
            max_num_passengers = int(np.max(nump))
            max_mode = max_num_passengers*2+2
            modes = np.empty(nump.shape,dtype=np.int_)
            modes_driving = nump*state + max_num_passengers+1 #modes for times when driving
            modes_idling = nump*np.logical_not(state_boolean) #modes for times when idling
            modes[state_boolean] = modes_driving[state_boolean] #selecting right modes with mask
            modes[np.logical_not(state_boolean)] = modes_idling[np.logical_not(state_boolean)]
            time_durations = np.diff(np.array(self._timepoints))
            max_mode = int(np.max(modes))
            num_vehicles = modes.shape[0]
            # assert that this makes sense
            assert num_vehicles == len(self._vehicle_id_index),\
                "Yeah, go over your code! Good thing it is a minefield of assertions."
            # assert np.all(modes[:,-1] == 0), \
            #     "Uuups, terminal state should be idling. Actually can allow this if doing live plot."
            # calculate
            result = np.zeros((num_vehicles,max_mode+1))
            for vehicle_id in range(0,num_vehicles):
                for mode in range(0,max_mode+1):
                    mask = modes[vehicle_id,0:-1]  == mode #ignore last element as it is the terminal state
                    if not np.any(mask): #if never in this mode, do not bother counting
                        continue
                    result[vehicle_id,mode] = np.sum(time_durations[mask])
            # assert result: time in all states needs to be the same, allow for some numerical instabilities
            sum_over_vehicles = np.sum(result,1)
            assert np.all(sum_over_vehicles <= sum_over_vehicles[0] + 1e-5) and \
                np.all(sum_over_vehicles >= sum_over_vehicles[0] - 1e-5), \
                f"Wait, vehicles have different total times in all modes: {sum_over_vehicles}. And you thought this code is so nice and nifty..."
            return result
        else:
            raise NotImplementedError(f"Did not implement getting data from the vehcile metrics with the identifier {identifier}")

class CalculationTimeRecord:
    """
    Record of a single calculation time entry

    Note: opted for dictionary as could also add different values
    
    Attributes
    ----------
    record: np.ndarray
        array to store the records with the following indexes
            0 ... time in s that it took the controller to compute (real time)
            1 ... time in s where in simulation this computation was triggered
            2 ... time in s when this was computed (with 0 being start of simulation)
    """
    def __init__(self,
                 computation_time: float,
                 simulation_timepoint: float,
                 real_timepoint: float) -> None:
        self.record: np.ndarray = np.array([
            computation_time,
            simulation_timepoint,
            real_timepoint],
            dtype=np.float64)
    def get_real_timepoint(self):
        return self.record[2]

class CalculationTimeMetrics:
    """
    Class to store and handle calucaltion times

    Attributes
    ----------
    _storage_controller_times: list[CalculationTimeRecord
        (private) storing the times it took for the controller to calculate
    _storage_env_times: list[CalculationTimeRecord]
        (private) storing the times it took for the environment to calculate
    _storage_matrix_controller_cache: np.ndarray
        (private) cache of matrix
    _storage_matrix_env_cache: np.ndarray
        (private) cache of matrix
    """
    def __init__(self):
        self._storage_controller_times: list[CalculationTimeRecord] = []
        self._storage_env_times: list[CalculationTimeRecord] = []
        self._storage_matrix_controller_cache: np.ndarray = None
        self._storage_matrix_env_cache: np.ndarray = None

    def __str__(self):
        # get data
        cputime_ctr = self.get_time_data(DataIdentifier.CALCULATION_CONTROLLER_TIMES)
        cputime_env = self.get_time_data(DataIdentifier.CALCULATION_ENVIRONMENT_TIMES)
        # formulate string
        return f"Computation Time Data:\n"+\
            f"Controller Computation Time:\n"+\
            f"\t\tmean: {np.mean(cputime_ctr):.3f}s\n"+\
            f"\t\tmax:  {np.max(cputime_ctr):.3f}s\n"+\
            f"Environment Computation Time:\n"+\
            f"\t\tmean: {np.mean(cputime_env):.3f}s\n"+\
            f"\t\tmax:  {np.max(cputime_env):.3f}s\n"
    def __repr__(self):
        return str(self)

    def add_controller_record(self,
                              record = CalculationTimeRecord):
        # ensure that monotone
        assert len(self._storage_controller_times) == 0 \
            or self._storage_controller_times[-1].get_real_timepoint() <= record.get_real_timepoint(), \
            "Stop violating the 2nd law of thermodynamics!"
        # add
        self._storage_controller_times.append(record)
    def add_env_record(self,
                       record = CalculationTimeRecord):
        # ensure that monotone
        assert len(self._storage_env_times) == 0 \
            or self._storage_env_times[-1].get_real_timepoint() <= record.get_real_timepoint(), \
            "Stop violating the 2nd law of thermodynamics!"
        # add
        self._storage_env_times.append(record)

    def get_time_data(self,
                      identifier: DataIdentifier) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Obtain the data from the calcualtion time metric according to the indentifier

        Returns the values in tuples (if needed)
        """
        if identifier == DataIdentifier.CALCULATION_CONTROLLER_TIME_AT_SIM_TIME:
            data = self._get_matrix_controller()
            return (data[1,:],data[0,:])
        elif identifier == DataIdentifier.CALCULATION_CONTROLLER_TIME_AT_REAL_TIME:
            data = self._get_matrix_controller()
            return (data[2,:],data[0,:])
        elif identifier == DataIdentifier.CALCULATION_CONTROLLER_REALTIME_VS_SIMTIME:
            data = self._get_matrix_controller()
            return (data[2,:],data[1,:])
        elif identifier == DataIdentifier.CALCULATION_CONTROLLER_TIMES:
            data = self._get_matrix_controller()
            return data[0,:]
        elif identifier == DataIdentifier.CALCULATION_ENVIRONMENT_TIME_AT_SIM_TIME:
            data = self._get_matrix_env()
            return (data[1,:],data[0,:])
        elif identifier == DataIdentifier.CALCULATION_ENVIRONMENT_TIME_AT_REAL_TIME:
            data = self._get_matrix_env()
            return (data[2,:],data[0,:])
        elif identifier == DataIdentifier.CALCULATION_ENVIRONMENT_REALTIME_VS_SIMTIME:
            data = self._get_matrix_env()
            return (data[2,:],data[1,:])
        elif identifier == DataIdentifier.CALCULATION_ENVIRONMENT_TIMES:
            data = self._get_matrix_env()
            return data[0,:]
        else:
            raise NotImplementedError(f"This is not the function you are looking for. This function cannot serve you the data {identifier}.")

    ### PRIVATE

    def _get_matrix_controller(self) -> np.ndarray:
        if self._storage_matrix_controller_cache is None \
            or self._storage_matrix_controller_cache.shape[1] != len(self._storage_controller_times):
            #rebuild
            matrix = np.empty((3,len(self._storage_controller_times)),
                              dtype=np.float64)
            for i,record in enumerate(self._storage_controller_times):
                matrix[:,i] = record.record
            self._storage_matrix_controller_cache = matrix
        return self._storage_matrix_controller_cache
    def _get_matrix_env(self) -> np.ndarray:
            if self._storage_matrix_env_cache is None \
                or self._storage_matrix_env_cache.shape[1] != len(self._storage_env_times):
                #rebuild
                matrix = np.empty((3,len(self._storage_env_times)),
                                dtype=np.float64)
                for i,record in enumerate(self._storage_env_times):
                    matrix[:,i] = record.record
                self._storage_matrix_env_cache = matrix
            return self._storage_matrix_env_cache

class MetricPlot:
    """
    Handles plots for metrics

    Attributes
    ----------
    _passenger_metrics: PassengerMetrics
        (private) reference to the passengermetrics (access via MetricMaster!)
    _vehicle_metrics: VehicleMetrics
        (private) reference to teh vehicle metrics (access via MetricMaster!)
    _calcualtion_metrics: CalculationTimeMetrics
        (private) reference to the calculation time metrics (access via MetricMaster!)
    _fig_storage: dict[str,Figure]
        (private) storage of figures (to save them after)
    """

    def __init__(self,
                 passenger_metrics: PassengerMetrics,
                 vehicle_metrics: VehicleMetrics,
                 calculation_metrics: CalculationTimeMetrics):
        # set up data references
        self._passenger_metrics: PassengerMetrics = passenger_metrics
        self._vehicle_metrics: VehicleMetrics = vehicle_metrics
        self._calcualtion_metrics: CalculationTimeMetrics = calculation_metrics
        # plt.style.use(['science','ieee'])
        self._fig_storage: dict[str,Figure] = dict()

    def plot_passenger_metrics(self,
                               show_plot = False):
        """
        creates and plots a figure with passenger results
        """
        # plt.style.use(['science','ieee'])
        data = self._passenger_metrics.get_passenger_data(
            DataIdentifier.PASSENGER_RECORDED_WAITING_TIME)
        numPassangers = len(data)
        # create figure
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.canvas.manager.set_window_title(f"Environment Passenger Metrics (Passangers Served: {numPassangers})")
        fig.suptitle(f"Environment Passenger Metrics (Passangers Served: {numPassangers})", fontsize=16)
        self._fig_storage['passenger_metrics'] = fig
        # ax_ridetime: Axes = axes[0]
        # ax_waitingtime: Axes = axes[1]
        ax_ridetime: Axes = axes[0, 0]
        ax_waitingtime: Axes = axes[0, 1]
        ax_optimalitynowaittime: Axes = axes[1, 0]
        ax_optimalitywaittime: Axes = axes[1, 1]

        # plot different axis
        ax = ax_ridetime
        data = self._passenger_metrics.get_passenger_data(
            DataIdentifier.PASSENGER_RECORDED_TRIP_DURATION)
        
        ax.set_title("Passenger Ride Time")
        ax.set_xlabel("Ride Time in Minutes")
        ax.set_ylabel("Number of Trips")
        ax.grid(True)
        ax.xaxis.set_major_formatter(FuncFormatter(FormatHelper.format_duration_min))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(60*10)) #grid every 10min
        #plot data
        sns.histplot(data=data,ax=ax,
                     stat='count',
                     kde=True)
        # sns.rugplot(data=data,ax=ax)
        #add mean
        mean_data = np.mean(data)
        ax.axvline(
            mean_data,
            color='red', linestyle='--', linewidth=1,
            label=f'Mean = {FormatHelper.format_duration_min_s(mean_data)}'
        )
        ax.legend()

        ax = ax_waitingtime
        data = self._passenger_metrics.get_passenger_data(
            DataIdentifier.PASSENGER_RECORDED_WAITING_TIME)
        ax.set_title("Passenger Waiting Time")
        ax.set_xlabel("Waiting Time in Minutes")
        ax.set_ylabel("Number of Trips")
        ax.grid(True)
        ax.xaxis.set_major_formatter(FuncFormatter(FormatHelper.format_duration_min))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(60*5)) #grid every 5 minutes
        #plot data
        sns.histplot(data=data,ax=ax,
                     stat='count',
                     kde=True)
        # sns.rugplot(data=data,ax=ax)
        #add mean
        mean_data = np.mean(data)
        ax.axvline(
            mean_data,
            color='red', linestyle='--', linewidth=1,
            label=f'Mean = {FormatHelper.format_duration_min_s(mean_data)}'
        )
        ax.legend()

        ax = ax_optimalitynowaittime
        data = self._passenger_metrics.get_passenger_data(
            DataIdentifier.PASSENGER_TRIP_OPTIMALITY_NO_WAITING_TIME)
        ax.set_title("Trip Optimality without Waiting Time")
        ax.set_xlabel("Optimality $\\frac{T}{\min T}$")
        ax.set_ylabel("Density")
        ax.grid(True)
        #plot data
        if not (np.all(data <= 1.0+1e-5) and np.all(data >= 1.0-1e-5)):
            sns.histplot(data=data,ax=ax,
                         stat='density',
                         kde=True)
            sns.rugplot(data=data,ax=ax)
        else:
            ax.plot([1, 1], [1, 0], color=sns.color_palette()[0])
            ax.plot([1, 3], [0, 0], color=sns.color_palette()[0])
            ax.text(2, 0.05, "All Optimal", ha='center', va='bottom', fontsize=12)
            sns.rugplot(data=data,ax=ax)
        #add mean
        mean_data = np.mean(data)
        ax.axvline(
            mean_data,
            color='red', linestyle='--', linewidth=1,
            label=f'Mean = {mean_data:.2f}'
        )
        ax.legend()
        ax.set_ylim(0,1)

        ax = ax_optimalitywaittime
        data = self._passenger_metrics.get_passenger_data(
            DataIdentifier.PASSENGER_TRIP_OPTIMALITY_WITH_WAITING_TIME)
        ax.set_title("Trip Optimality with Waiting Time")
        ax.set_xlabel("Optimality $\\frac{T}{\min T}$")
        ax.set_ylabel("Density")
        ax.grid(True)
        #plot data
        sns.histplot(data=data,ax=ax,
                     stat='density',
                     kde=True)
        sns.rugplot(data=data,ax=ax)
        #add mean
        mean_data = np.mean(data)
        ax.axvline(
            mean_data,
            color='red', linestyle='--', linewidth=1,
            label=f'Mean = {mean_data:.2f}'
        )
        ax.legend()
        ax.set_ylim(0,1)

        fig.tight_layout() #fix overlaps

        # show plot
        if show_plot:
            plt.show(block=False)
            plt.pause(0.01) #to ensure that shown

    def plot_vehicle_metrics(self,
                             show_plot = False):
        """
        create and plots a figure with vehicle metrics
        """

        # create figure
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title("Environment Vehicle Metrics")
        fig.suptitle("Environment Vehicle Metrics", fontsize=16)
        self._fig_storage['vehicle_metrics'] = fig #store figure

        data = self._vehicle_metrics.get_vehicle_data( #load data here to build ratio
            identifier=DataIdentifier.VEHCILE_TIME_IN_MODE_PER_VEHICLE)
        num_vehicles = data.shape[0]
        num_mode = data.shape[1]
        assert num_mode % 2 == 0, \
            "You have the same amount of modes for driving and for idling - this must be an even number!"
        num_max_pasenger = num_mode//2-1

        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, num_vehicles])
        ax_mode_avg = fig.add_subplot(gs[0, 0])
        ax_mode_individual = fig.add_subplot(gs[0, 1], sharey=ax_mode_avg)
        ax_num_pandv = fig.add_subplot(gs[1, :])
        ax_pinv = fig.add_subplot(gs[2, :], sharex=ax_num_pandv)

        # do indivudal plots
        ax = ax_mode_avg
        # prepare data into dataframe to do it easier with seaborne (ChatGPT helped here a lot)
        labels = [verb + f" with {i} Passenger{'s' if i != 1 else ''}" for verb in ["Idling", "Driving"] for i in range(0,num_max_pasenger+1)]
        df = pd.DataFrame(data,
                          columns=labels)
        df["Vehicle"] = [f"Vehicle {i+1}" for i in range(df.shape[0])]
        df_long = df.melt(id_vars="Vehicle", var_name="Feature", value_name="Value")
        # do plot
        mean_df = df_long.groupby("Feature")["Value"].mean().reset_index()
        bottom = 0
        colors = sns.color_palette("pastel", len(mean_df))
        for i, row in mean_df.iterrows():
            ax.bar(0, row["Value"], bottom=bottom, color=colors[i], label=row["Feature"])
            bottom += row["Value"]
        ax.set_title("Average Vehicle Use")
        ax.set_ylabel("Simulation Time in h")
        ax.yaxis.set_major_locator(MultipleLocator(60*60*4)) #every 4h a line
        ax.yaxis.set_major_formatter(FuncFormatter(FormatHelper.format_duration_h))
        ax.set_xticks([0])
        ax.set_xticklabels(["Total"])
        ax.grid(axis='y')
        # ax.legend() #other plot will have same legend

        ax = ax_mode_individual
        pivot_df = df_long.pivot(index="Vehicle", columns="Feature", values="Value").fillna(0)
        bottom = np.zeros(len(pivot_df))
        x = np.arange(len(pivot_df))
        for i, col in enumerate(pivot_df.columns):
            ax.bar(x, pivot_df[col], bottom=bottom, label=col, color=colors[i])
            bottom += pivot_df[col].values
        ax.set_title("Indivudal Vehicle Use")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.set_ylabel("Simulation Time in h")
        ax.yaxis.set_major_locator(MultipleLocator(60*60*4))
        ax.yaxis.set_major_formatter(FuncFormatter(FormatHelper.format_duration_h))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.grid(axis='y')
        ax.legend()

        ax = ax_num_pandv
        ax_r = ax_num_pandv.twinx()
        colors = sns.color_palette("muted", n_colors=2) #select colors from seaborne
        color_p = colors[0]
        color_v = colors[1]
        #number of passengers riding
        time, data = self._vehicle_metrics.get_vehicle_time_data(
            identifier=DataIdentifier.VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_TOTAL)
        ax.step(x=time,y=data,where='post',
                color=color_p, label="#Passengers Riding",linestyle='-')
        #number of passengers waiting
        time, data = self._vehicle_metrics.get_vehicle_time_data(
            identifier=DataIdentifier.LOCATION_WAITING_PASSENGERS_AT_TIME_TOTAL)
        ax.step(x=time,y=data,where='post',
                color=color_p, label="#Passengers Waiting",linestyle='--')
        #number of vehicles driving
        time, data = self._vehicle_metrics.get_vehicle_time_data(
            identifier=DataIdentifier.VEHICLE_NUM_VEHICLE_DRIVING_AT_TIME)
        ax_r.step(x=time,y=data,where='post',
                color=color_v, label="#Vehicles Driving",linestyle='-')
        #number of vehicles idling
        time, data = self._vehicle_metrics.get_vehicle_time_data(
            identifier=DataIdentifier.VEHICLE_NUM_VEHICLE_IDLING_AT_TIME)
        ax_r.step(x=time,y=data,where='post',
                color=color_v, label="#Vehicles Idling",linestyle='--')
        ax.set_title("Number of Passengers and Vehicles Over Simualtion Time")
        self._set_axis_for_simtime(ax)
        ax.set_ylabel("Number of Passengers")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, color=color_p, linestyle=':')
        ax_r.set_ylabel("Number of Vehicles")
        ax_r.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_r.yaxis.grid(True, color=color_v, linestyle=':')
        ax.legend()
        ax_r.legend()

        ax = ax_pinv
        time, data = self._vehicle_metrics.get_vehicle_time_data(
            identifier=DataIdentifier.VEHICLE_NUM_PASSENGER_IN_VEHICLE_AT_TIME_PER_VEHICLE)
        num_vehicles = data.shape[0]
        colors = sns.color_palette("muted", n_colors=num_vehicles) #select colors from seaborne
        for i in range(num_vehicles):
            ax.step(x=time,y=data[i,:],where='post',
                    color=colors[i],linewidth=0.5, label=f"Vehicle {i+1}")
        ax.step(x=time,y=np.mean(data,0),where='post',
                color='red',linewidth=1.5, label=f"Mean")
        ax.set_title("Number of Passengers in Vehicles Over Simulation Time")
        self._set_axis_for_simtime(ax)
        ax.xaxis.grid(True, which='both')
        ax.set_ylabel("Number of Passengers")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, linestyle=':')
        ax.legend()

        fig.tight_layout() #fix overlaps

        # show plot
        if show_plot:
            plt.show(block=False)
            plt.pause(0.01) #to ensure that shown

    def plot_calculation_metrics(self,
                                 show_plot=False):
        """plot the calcualtion metrics"""

        # create figure
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title("Calculation Time Metrics")
        fig.suptitle("Calculation Time Metrics", fontsize=16)
        self._fig_storage['calculation_time_metrics'] = fig #store figure

        # set up axes
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
        ax_simetime = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_realtime = fig.add_subplot(gs[1, 0])
        ax_realtionship = fig.add_subplot(gs[1, 1])

        # set up colors
        colors = sns.color_palette("muted", n_colors=2) #select colors from seaborne
        color_ctr = colors[0]
        color_env = colors[1]

        # plot individual axes
        ax = ax_simetime
        ax_r = ax_simetime.twinx()
        simtime,cputime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_CONTROLLER_TIME_AT_SIM_TIME)
        sns.lineplot(x=simtime,y=cputime,ax=ax,
                     label="Controller", color=color_ctr)
        simtime,cputime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_ENVIRONMENT_TIME_AT_SIM_TIME)
        sns.lineplot(x=simtime,y=cputime,ax=ax,
                     label="Environment", color=color_env)
        self._set_axis_for_simtime(ax)
        self._set_axis_for_computationtime(ax)
        self._set_axis_for_computationtime(ax_r)
        ax.yaxis.grid(True, color=color_ctr, linestyle=':')
        ax_r.yaxis.grid(True, color=color_env, linestyle=':')
        ax.set_title("Computation Time vs. Simulation Time")
        ax.legend()

        ax = ax_realtime
        ax_r = ax_realtime.twinx()
        realtime,cputime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_CONTROLLER_TIME_AT_REAL_TIME)
        sns.lineplot(x=realtime,y=cputime,ax=ax,
                     label="Controller", color=color_ctr)
        realtime,cputime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_ENVIRONMENT_TIME_AT_REAL_TIME)
        sns.lineplot(x=realtime,y=cputime,ax=ax,
                     label="Environment", color=color_env)
        self._set_xaxis_for_realtime(ax)
        self._set_axis_for_computationtime(ax)
        self._set_axis_for_computationtime(ax_r)
        ax.yaxis.grid(True, color=color_ctr, linestyle=':')
        ax_r.yaxis.grid(True, color=color_env, linestyle=':')
        ax.set_title("Computation Time vs. Real Time")
        ax.legend()

        ax = ax_hist
        ax.set_box_aspect(1) #make this a perfect square s.t. can see directly if it is 45deg
        cputime_ctr = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_CONTROLLER_TIMES)
        sns.histplot(data=cputime_ctr,ax=ax,
                     stat='density',
                     kde=True,
                     label="Controller",color=color_ctr)
        sns.rugplot(data=cputime_ctr,ax=ax,color=color_ctr)

        ax.set_ylabel("Density")
        self._set_axis_for_computationtime(ax=ax,which='x')
        ax.set_title("Spread of Computation Times")
        ax.legend()

        ax = ax_realtionship
        ax.set_title("Relationship Real and Simulation Time")
        realtime,simtime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_CONTROLLER_REALTIME_VS_SIMTIME)
        sns.scatterplot(x=realtime,y=simtime,ax=ax,
                        label="Controller",color=color_ctr)
        realtime,simtime = self._calcualtion_metrics.get_time_data(
            identifier=DataIdentifier.CALCULATION_ENVIRONMENT_REALTIME_VS_SIMTIME)
        sns.scatterplot(x=realtime,y=simtime,ax=ax,
                        label="Environment",color=color_env, marker="x")
        self._set_xaxis_for_realtime(ax)
        self._set_axis_for_simtime(ax,which='y')
        ax.legend()

        fig.tight_layout() #fix overlaps

        # show plot
        if show_plot:
            plt.show(block=False)
            plt.pause(0.01) #to ensure that shown

    def save_figure(self,outname:str = None):
        """
        Save all figures to disk
        """
        generate_outname = False
        if outname:
            outname = outname
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            generate_outname = True

        for name,fig in self._fig_storage.items():
            for format in ['png']:
                if generate_outname:
                    outname = f"environment-plot__{name}__{timestamp}"
                fname = f"out/{outname}.{format}"
                fig.savefig(fname=fname,
                    format=format)
                print(f"Saved Plot \"{name}\" as an {format} at {fname}.")

    ### PRIVATE
    @staticmethod
    def _set_axis_for_simtime(ax: Axes,
                              which='x'):
        if which=='x':
            ax.set_xlabel("Simulation Time in D HH:MM")
            axis = ax.xaxis
        else:
            ax.set_ylabel("Simulation Time in D HH:MM")
            axis = ax.yaxis
        axis.set_major_formatter(FuncFormatter(FormatHelper.format_time))
        axis.set_major_locator(MultipleLocator(60*60*4)) #major line every 4 hours
        axis.set_minor_locator(MultipleLocator(60*60)) #minor line every h
        axis.grid(True, which='both')
    @staticmethod
    def _set_xaxis_for_realtime(ax: Axes):
        ax.set_xlabel("Real Time in H:MM:SS")
        ax.xaxis.set_major_formatter(FuncFormatter(FormatHelper.format_time_precise))
        ax.xaxis.set_major_locator(MultipleLocator(60*5)) #major line every 5 min
        ax.xaxis.set_minor_locator(MultipleLocator(60)) #major line every min
        ax.xaxis.grid(True, which='both')
    @staticmethod
    def _set_axis_for_computationtime(ax: Axes,
                                      which:str='y'):
        if which=='y':
            ax.set_ylabel("Computation Time in s")
            axis=ax.xaxis
        else:
            ax.set_xlabel("Computation Time in s")
            axis=ax.yaxis
        axis.grid(True)

class MetricMaster:
    """
    Managing all the metrics. 
    Use to record data or visualize data.

    Attributes
    ----------
    passenger_metrics: PassengerMetrics
        Metrics of passengers
    vehicle_metrics: VehicleMetrics
        Metrics of vehicles
    calcualtion_metrics: CalculationTimeMetrics
        Netrics of calculation times

    _geo: Geography
        (private) Geograpy object to do some trajectory calcualtions
    _metric_plot: MetricPlot
        (private) Plot object to do the plots with
    """
    DEBUG_PRINT = False

    def __init__(self,
                 vehicle_ids: set[uuid.UUID],
                 geo: Geography):
        self._geo = geo
        
        self.passenger_metrics: PassengerMetrics = PassengerMetrics()
        self.vehicle_metrics: VehicleMetrics = VehicleMetrics(vehicle_ids=vehicle_ids)
        self.calcualtion_metrics: CalculationTimeMetrics = CalculationTimeMetrics()

        self._metric_plot: MetricPlot = MetricPlot(
            passenger_metrics=self.passenger_metrics,
            vehicle_metrics=self.vehicle_metrics,
            calculation_metrics=self.calcualtion_metrics
        )

    def __str__(self):
        #get data
        time,_ = self.vehicle_metrics.get_vehicle_time_data(
            DataIdentifier.VEHICLE_NUM_PASSENGER_RIDING_AT_TIME_TOTAL)
        simulation_duration = time[-1]-time[1]
        #construct string
        return f"General Data:\n"+\
            f"\tSimulated Duration: {FormatHelper.format_time_duration(simulation_duration)}\n"+\
            f"{self.passenger_metrics}{self.vehicle_metrics}{self.calcualtion_metrics}"
    def __repr__(self):
        return str(self)
  
    def extract_passenger(self,
                          passenger: Passenger):
        """
        Extract all information of a passenger object and stores them locally

        Note: only makes sense if this passenger is removed after and not extracted again
        """
        if MetricMaster.DEBUG_PRINT:
            MetricMaster._debug_print(
                f"Extracting information of passenger {passenger.id}..."
            )
        # caluclate minimal time
        min_time = self._geo.predict_travel_duration(
            departure_location=passenger.location_pickup,
            arrival_location=passenger.location_dropoff
        )

        # add record
        self.passenger_metrics.add_records(
            record=PassengerRecord(
                passenger=passenger,
                minimal_journey_time_s=min_time.duration_s
            ),
            passenger_id=passenger.id
        )

    def callback_vehicle_stateupdate(self,
                                     time: SimTime,
                                     vehicle: Vehicle,
                                     old_state: VehicleState):
        """callback function that needs to be called from the 
        env loop whenever there is a state change"""
        self.vehicle_metrics.add_record(
            time_s = time.time_s,
            state_old=old_state,
            state_new=vehicle.state,
            num_p_new=len(vehicle.passengers),
            num_p_old=len(vehicle.passengers),
            vehicle_id=vehicle.id
        )

    def callback_vehicle_passengerchange(self,
                                         time: SimTime,
                                         vehicle: Vehicle,
                                         old_num_passenger: int):
        """callback function that needs to be called from the 
        env loop whenever the passengers in a vehicle change"""
        self.vehicle_metrics.add_record(
            time_s = time.time_s,
            state_old=vehicle.state,
            state_new=vehicle.state,
            num_p_new=len(vehicle.passengers),
            num_p_old=old_num_passenger,
            vehicle_id=vehicle.id
        )

    def callback_waiting_passengerchange(self,
                                         time: SimTime,
                                         old_at_location: int,
                                         new_at_location: int):
        """callback function that needs to be called from the 
        env loop whenver the amount of waiting passengers at a location changes"""
        self.vehicle_metrics.add_record_waiting(
            time_s=time.time_s,
            delta=new_at_location-old_at_location
        )

    def register_controller_computation(self,
                                        computation_time_s: float,
                                        simulation_timepoint: SimTime,
                                        realtime_timepoint_s: float):
        """
        Register how long a computation of the controller took

        Parameters
        ----------
        computation_time_s: float
            computation time of the controller in s (real time)
        simulation_timepoint: SimTime
            timepoint when the computation _started_ in the simulation
        realtime_timepoint_s: float
            timepoint when the computation _started_ in real time (with 0 being the start of the simulation)
        """
        self.calcualtion_metrics.add_controller_record(
            CalculationTimeRecord(
                computation_time=computation_time_s,
                simulation_timepoint=simulation_timepoint.time_s,
                real_timepoint=realtime_timepoint_s
            )
        )
    def register_env_computation(self,
                                        computation_time_s: float,
                                        simulation_timepoint: SimTime,
                                        realtime_timepoint_s: float):
        """
        Register how long a computation of the environment took

        Parameters
        ----------
        computation_time_s: float
            computation time of the environment in s (real time)
        simulation_timepoint: SimTime
            timepoint when the computation _started_ in the simulation
        realtime_timepoint_s: float
            timepoint when the computation _started_ in real time (with 0 being the start of the simulation)
        """
        self.calcualtion_metrics.add_env_record(
            CalculationTimeRecord(
                computation_time=computation_time_s,
                simulation_timepoint=simulation_timepoint.time_s,
                real_timepoint=realtime_timepoint_s
            )
        )
        
    def plot(self):
        """
        Create all plots
        """
        if MetricMaster.DEBUG_PRINT:
            MetricMaster._debug_print("Plotting collected Environment Metrics...")
        self._metric_plot.plot_passenger_metrics()
        self._metric_plot.plot_vehicle_metrics()
        self._metric_plot.plot_calculation_metrics(show_plot=True)

        #save
        self._metric_plot.save_figure()

        #show
        plt.show(block=False)
        input("Showing Plot of Environment Metrics. Press ENTER to continue.")
    def save_metrics(self,filename: str = "metrics"):
        """Disclaimer: does store underlying geography as well, hence resulting in huge files."""
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        FILE_NAME = filename + timestamp + ".pkl"
        with open(FILE_NAME, 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
    # STILL TODO from old metric class
    #         print("Vehicle Metrics:")
    #         print(f"  Average Number of Rides: {self._vehicle_metrics['average_number_of_rides']:.2f}")
    #         print(f"  Average Travel Time: {self._vehicle_metrics['average_travel_time']:.2f}s")
    
    ### DEBUG ###
    def _debug_print(str: str):
        print("[METRIC] "+str)