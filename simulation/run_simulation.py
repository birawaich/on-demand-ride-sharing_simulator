import time
import pickle as pkl
import os
import matplotlib.pyplot as plt
from metrics import FormatHelper

from environment import Environment
from geo import Geography
from bielegrid_geography import BieleGridGeography
from man_geography import ManGridGeography
from visualization import EnvironmentVisualization

from bielegrid import BieleGrid
from mangrid import ManGrid
from randomtaxicontroller import RandomTaxiController
from nearesttaxicontroller import NearestTaxiController
from metrics import MetricMaster
from multiplepassengercontroller import MultiplePassengerController
from indecisivecontroller import IndecisiveController
from noveltaxicontroller import NovelTaxiController
from rebalancing_controller import CombinedController, RebalancingController

### SETTINGS

sim_duration_s = 24*60*60 #simulation duration in simulation seconds

Environment.DEBUG_PRINT = True
Environment.SAVE_ENVIRONMENT = False
Environment.LOAD_ENVIRONMENT = False #see further down, loads a manhatten grid
Environment.DO_VISUALIZATION = False
Environment.DO_PLOT = True
BieleGrid.DEBUG_PRINT = True
BieleGridGeography.DEBUG_PRINT = False
ManGrid.DEBUG_PRINT = True
ManGridGeography.DEBUG_PRINT = False
EnvironmentVisualization.DEBUG_PRINT = False
EnvironmentVisualization.DO_LIVE_VISUALIZATION = False
Geography.DEBUG_PRINT = True

RebalancingController.DEBUG_PRINT = True
RebalancingController.DO_REBALANCING = True #set to False to disable rebalancing but still collect spread information
RebalancingController.PLOT_REGIONS = False
RebalancingController.SAVE_REGION_PLOT = False
CombinedController.STORE_REBALANCINGMETRICS = True

NUM_VEHICLES = 2000 #only for manhattan environment, see below
VEHICLE_CAPACITY = 2 #only for manhattan environment, see below
### END SETTINGS

# set up environment
if Environment.LOAD_ENVIRONMENT:
    dir_path = os.path.dirname(os.path.realpath(__file__)) #get current directory
    print("loading")
    env:ManGrid = pkl.load(open(dir_path + "/../../data/ManGrid_env_day.pkl", "rb"))
    plt.close('all') #close all plots, so that the environment can be visualized
    print("reinit")
    env.re_init(reset_vehicles=True, num_vehicles= NUM_VEHICLES, vehicle_capacity = VEHICLE_CAPACITY)
else:
    # env = ManGrid(sim_duration_s=sim_duration_s)
    env = BieleGrid(sim_duration_s=sim_duration_s)

print("Done Setup Environment.")
print(f"Number of total passengers: {len(env._passengers)}")
# set up controller
# controller = RandomTaxiController(
#     locations=env.get_location_ids(),
#     vehicles=env.get_vehicles_ids()
# )
# controller = NearestTaxiController(
#     locations=env.get_location_ids(),
#     vehicles=env.get_vehicles_ids(),
#     geo=env.get_geography(),
#     env=env
# )
# baseline_controller = MultiplePassengerController(
#     locations=env.get_location_ids(),
#     vehicles=env.get_vehicles_ids(),
#     geo=env.get_geography(),
#     env=env
# )

novel_controller = NovelTaxiController(
    locations= env.get_location_ids(),
    vehicles=env.get_vehicles_ids(),
    geo=env.get_geography(),
    env=env
)
controller = CombinedController(
    main_controller=novel_controller, #change here to switch controller ;)
    env=env,
    num_regions_min=5 #the value of 20 was used for the manhatten grid
)
novel_controller.set_communication_range(controller._rebalancing_controller._T_trip)

# run simulation
start_simualtion = time.perf_counter()
start_env = start_simualtion
i = 0
while(1):
    # get event from environment OR next control time
    event = env.next_event()
    end_env = time.perf_counter()
    if i % 1000 == 0:
        print(event.timepoint.get_humanreadable())
        i = 0
    #terminate once environment there is no more observations
    if event is None: 
        break

    # register time
    env.metric_master.register_env_computation(
        computation_time_s=end_env-start_env,
        simulation_timepoint=event.timepoint,
        realtime_timepoint_s=start_env-start_simualtion
    )

    # send to controller

    start_controller = time.perf_counter()
    action = controller.process_event(event)
    # action = novel_controller.process_event(event)
    end_controller = time.perf_counter()
    # register time
    env.metric_master.register_controller_computation(
        computation_time_s=end_controller-start_controller,
        simulation_timepoint=event.timepoint,
        realtime_timepoint_s=start_controller-start_simualtion
    )

    # give action back to environment
    start_env = time.perf_counter()
    env.register_action(action=action)
    i = i + 1

end_simulation = time.perf_counter()
duration = end_simulation-start_simualtion
print(f"Done Running Simulation in {FormatHelper.format_time_duration(duration)}"+\
      f", ie. {sim_duration_s/duration:.1f} faster than real time.")

# uncomment to store metrics
# if RebalancingController.DO_REBALANCING:
#     print(f"Run completed WITH rebalancing controller")
#     env.metric_master.save_metrics(filename=str(VEHICLE_CAPACITY) + "-" + str(NUM_VEHICLES) +"RebalancingMetrics")
# else:
#     print(f"Run completed WITHOUT rebalancing controller")
#     env.metric_master.save_metrics(filename=str(VEHICLE_CAPACITY) + "-" + str(NUM_VEHICLES) + "NoRebalancingMetrics")

# finalize
if type(controller) is CombinedController:
    controller.finalize()
env.finalize()