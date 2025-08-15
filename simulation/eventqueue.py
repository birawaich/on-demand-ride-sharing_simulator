import heapq
from dataclasses import dataclass, field

from event import Event, VisualizationRequest, ControllerTimingTick

@dataclass(order=True)
class PrioriatizedEvent:
    """Local Dataclass to insert stuff neatly into queue"""
    priority: float =field(compare=True)
    item: Event=field(compare=False)
    

class EventQueue:
    """
    Eventqueue for simulation

    Supports
    - Inserting events with some simtime (stored in Event)
    - Popping the closest event in time

    Uses heapq

    Attributes
    ----------
    _event_queue: heapq
        actual queue
    _num_concrete_events: int
        amount of non visualization or non timing events that are in queue
    """

    def __init__(self):
        self._event_queue = []
        # note: why PriorityQueueu and not heapq: would support thread safety
        # but we do not use threads, so we can use heapq directly and this allows saving of files
        self._num_concrete_events: int = 0

    def put(self,
            event: Event):
        """
        Insert an event into the event queue
        """
        if not isinstance(event,VisualizationRequest)\
            and not isinstance(event,ControllerTimingTick):
            self._num_concrete_events += 1
        heapq.heappush(
            self._event_queue,
            PrioriatizedEvent(
                priority=event.timepoint.time_s,
                item=event
            )
        )

    def get(self) -> Event:
        """
        Pops the next event in time; returns None if queue is empty
        """
        if len(self._event_queue) == 0:
            return None
        event = heapq.heappop(self._event_queue).item

        if not isinstance(event,VisualizationRequest)\
            and not isinstance(event,ControllerTimingTick):
            self._num_concrete_events -= 1

        return event
    
    def no_action_events_in_queue(self) -> bool:
        """
        Return true of there is no action events in queue
        i.e. there is only visualization events and controlelr timing ticks or no events
        """
        return self._num_concrete_events == 0