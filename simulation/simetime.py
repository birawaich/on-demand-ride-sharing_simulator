import datetime
import copy

class SimDuration:
    """
    Class to keep track of simulation time spans

    Again, to get rid of stupid errors with time

    Attributes
    ----------
    duration_s
        Duration in seconds
    """
    def __init__(self, duration_in_sec):
        self.duration_s = duration_in_sec

    def __repr__(self):
        return f"{int(self.duration_s//60)}min {round(self.duration_s % 60,1)}s"

class SimTime:
    """
    Class to keep track of simulation time
    
    ... basically to avoid "was this seconds?", "how do I get a HH:MM reading out of this?"

    Atttributes
    -----------
    time_s
        Time since simulation start in seconds
    """

    def __init__(self,time_in_sec):
        self.time_s: float = time_in_sec #store the time in seconds

    # comparision functions
    def __eq__(self, other):
        if isinstance(other, SimTime):
            return self.time_s == other.time_s
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, SimTime):
            return self.time_s != other.time_s
        return NotImplemented
    def __lt__(self, other):
        if isinstance(other, SimTime):
            return self.time_s < other.time_s
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, SimTime):
            return self.time_s > other.time_s
        return NotImplemented
    def __le__(self, other):
        if isinstance(other, SimTime):
            return self.time_s <= other.time_s
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, SimTime):
            return self.time_s >= other.time_s
        return NotImplemented
    
    # copy (deepcopy is not needed as there is just one foeld)
    def __copy__(self):
        return SimTime(time_in_sec=self.time_s)
    
    # string representations
    def __repr__(self):
        return self.get_humanreadable()
    def __str__(self):
        return self.get_humanreadable()

    def get_humanreadable(self) -> str:
        """
        Get a human readable clock format of the Sim Time
        """
        time_delta = datetime.timedelta(seconds=float(self.time_s))
        date = datetime.datetime(2000,1,1) + time_delta
        return date.strftime("%j %H:%M:%S")
    
    def add_duration(self,
                     duration: SimDuration):
        """
        Add a time duration to a time point and return result
        """
        self.time_s += duration.duration_s
        return self