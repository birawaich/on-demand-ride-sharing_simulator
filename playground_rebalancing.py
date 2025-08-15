"""Testing out stuff for the rebalancing controller"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
import scienceplots


def plot_directed_graph(G, layout='spring', node_size=500,
                        node_values=None):
    """
    Visualize a directed graph using NetworkX and Matplotlib.

    Created with ChatGPT
    
    Args:
        G: A NetworkX DiGraph
        layout: One of 'spring', 'circular', 'kamada_kawai'
        node_size: Size of the nodes in the plot
    """
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout '{layout}'")
    
    # Set labels
    if node_values is not None:
        labels = {node: f"({node}): {node_values[node]}" for node in G.nodes()}
    else:
        labels = {node: str(node) for node in G.nodes()}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='black')
    plt.title("Directed Graph (Strongly Connected)")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)

def generate_stochastic_matrix(n: int, p: float = None, fully_connected = False):
    """
    Generate an n x n row-stochastic matrix that corresponds to a 
    strongly connected directed graph.

    p is the probability that a verticy has an edge

    Generated with ChatGPT and adjusted
    """

    # Generate a strongly connected directed graph
    if p is None:
        p=1/n #initialize some probabiltiy
    if fully_connected:
        p=1
    while True:
        G = nx.gnp_random_graph(n, p, directed=False)
        G=G.to_directed()
        if nx.is_strongly_connected(G):
            break
        p += 0.01 #increase probabiltiy


    # Create stochastic matrix
    A = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G.successors(i))
        probs = np.ones(len(neighbors))
        probs /= probs.sum()  # Normalize to make row-stochastic
        for j, p in zip(neighbors, probs):
            A[i, j] = p

    return A, G

def demo_perfect_balancing():
    """Demo of the perfect balancing algorithm on vehicle positions as in the paper"""

    #     return res
    rng = np.random.default_rng(seed=42)

    N_iterations = 10000 #maximal value

    N_nodes = 100

    # A = np.zeros((4,4)) #toy example: graph with loop 1-2-3 and 4 connected only to 1 ... strongly connected
    # A[0,:] = 1/4
    # A[1,[0,1,2]] = 1/3
    # A[2,[0,1,2]] = 1/3
    # A[3,[0,3]] = 1/2
    # x = np.array([10,2,9,3])

    A,G = generate_stochastic_matrix(n=N_nodes)
    x = rng.integers(low=0,high=N_nodes**2,size=N_nodes)
    x_storage = np.empty((x.size,N_iterations))


    # print(A)
    # print(x)
    # print(A@x)

    # Gossiping Algorithms: Algorithm 1
    for iteration in range(N_iterations):
        print(f"iteration {iteration}:\t{x}\tmean: {np.mean(x)}\tsum: {np.sum(x)}")
        x_storage[:,iteration] = x

        # check if converged
        if np.max(x) - np.min(x) <=1:
            print(f"Have quantized consensus! ({iteration+1} Iterations for {N_nodes} Nodes)")
            N_iterations = iteration+1
            break

        # randomly select an edge
        while(True):
            vertices = rng.integers(low=0,high=x.size,size=2)
            if A[vertices[0],vertices[1]] != 0: #if they are connected, do something
                break

        # do update
        i = vertices[0]
        j = vertices[1]
        x_new = x.copy()
        x_new[i] = np.ceil(0.5*(x[i]+x[j]))
        x_new[j] = np.floor(0.5*(x[i]+x[j]))

        # in simulation: would now schedule rides from the source to the destination
        x = x_new

    # plot
    fig, ax = plt.subplots(1, 1,figsize=(8,6))
    iterations = range(N_iterations)
    for i in range(x_storage.shape[0]):
        ax.step(iterations, x_storage[i,:N_iterations], where='post',linewidth=0.5) #label=f"Node {i+1} #add this for node labling
    ax.plot(iterations, np.mean(x_storage[:,:N_iterations],axis=0), color='black', linewidth=1, label='Average')
    ax.set_xlabel("Iterations")
    ax.set_ylabel('Value')
    ax.set_title('Time Evolution of Values and Average')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    fig.savefig("out/rebalacing_testing.png", dpi=300)

def demo_gossip_algorithms():
    """Demo of algorithms as written down on 2025-07-18 by Benji"""

    ### SETTINGS
    N_iterations_max = 200 #maximal value
    N_nodes = 25
    N_vehicles = 111 #amount of vehilces in total
    vehicle_capacity = 4 #passenger per vehicle
    p_vehicle_empty = 0.2 #probability that a vehicle is empty ~> vehicles that can actually be moved

    selection_variants = ['multiple_uniform','multiple_delta_seat_L2'] #['multiple_uniform', 'single_uniform','multiple_delta_seat_L1','multiple_delta_seat_L2','multiple_delta_vehicle']
    update_variants = ['spots','spots_convergence'] #['vehicles_empty','spots','spots_convergence'] 

    N_experiments = 100 #amount of different experiements i.e. new graph ect.
    #behaves differently if set to 1 (plots!)
    CHECK_ASSUMPTION = True #check the assumption given in the proof of the paper, if True only let graphs pass that fullfill this assumption
    FULLY_CONNECTED_GRAPH = True #set to true to force the grtaph to be fully connected
    ### END SETTINGS

    rng = np.random.default_rng(seed=42)    

    ### LOCAL FUNCTIONS
    def select_edges_uniform(G,vehicles_empty) -> list[tuple[int,int]]:
        """Slect edges in a round uniformly"""
        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # assume uniform probability that the edge is selected; greedily select an edge, then remove all connecting edges of the vertex
        while(len(edges_with_vehicles)>0):
            edge = rng.choice(a=edges_with_vehicles,size=1,replace=False)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            edges_with_vehicles = [
                (i, j) for i, j in edges_with_vehicles
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]

        return selected_edges
    
    def select_edges_single_uniform(G,vehicles_empty) -> list[tuple[int,int]]:
        """Select a single edge (uniformly) from G that has at least one none empty vehicle"""
        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        selected_edges.append(rng.choice(a=edges_with_vehicles,size=1,replace=False)[0])
        return selected_edges
    
    def select_edges_delta_seats(G, vehicles_empty, spots, norm="L2") -> list[tuple[int,int]]:
        """Select edges where edges with a higher spot delta are more likely"""

        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # build probabilities
        probabilities = np.zeros(shape=len(edges_with_vehicles))
        for i, edge in enumerate(edges_with_vehicles):
            if norm == "L2":
                probabilities[i] = max((spots[edge[0]] - spots[edge[1]])**2,0.5) #ensure non-zero probability by counting a spot difference of 0 as 1/2 
            elif norm == "L1":
                probabilities[i] = max(np.abs(spots[edge[0]] - spots[edge[1]]),0.5)
            else:
                raise NotImplementedError(f"Unknown Norm {norm}")
        # draw until cannot draw anymore
        while(len(edges_with_vehicles)>0):
            probabilities = probabilities / np.sum(probabilities, dtype=float) #normalize, need to do at every step as elements are removed
            edge = rng.choice(a=edges_with_vehicles,size=1,replace=False,p=probabilities)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            index_allowed_edges = [
                index for index, (i,j) in enumerate(edges_with_vehicles)
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]
            probabilities = probabilities[index_allowed_edges]
            edges_with_vehicles = [edges_with_vehicles[i] for i in index_allowed_edges]
        return selected_edges
    
    def select_edges_delta_vehicles(G, vehicles_empty) -> list[tuple[int,int]]:
        """Select edges where edges with a higher spot delta are more likely"""

        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # build probabilities
        probabilities = np.zeros(shape=len(edges_with_vehicles))
        for i, edge in enumerate(edges_with_vehicles):
            probabilities[i] = max((vehicles_empty[edge[0]] - vehicles_empty[edge[1]])**2,0.5) #ensure non-zero probability by counting a spot difference of 0 as 1/2        
        # draw until cannot draw anymore
        while(len(edges_with_vehicles)>0):
            probabilities = probabilities / np.sum(probabilities, dtype=float) #normalize, need to do at every step as elements are removed
            edge = rng.choice(a=edges_with_vehicles,size=1,replace=False,p=probabilities)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            index_allowed_edges = [
                index for index, (i,j) in enumerate(edges_with_vehicles)
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]
            probabilities = probabilities[index_allowed_edges]
            edges_with_vehicles = [edges_with_vehicles[i] for i in index_allowed_edges]
        return selected_edges


    def update_edges_spots(spots,vehicles_empty,edges, convergence_guarantee: bool= False):
        """Update an edge selection aka. move vehciles based on spots values
        
        modifies the arrays spots and vehicles in palce"""
        for edge in edges:
            i = edge[0]
            j = edge[1]

            delta_s = np.abs(spots[i] - spots[j])

            # do action
            if delta_s == 0: #equal values # Ensures (P1)
                # print(f"Edge ({i},{j}) is already balanced in terms of seats!")
                continue

            if spots[i] > spots[j]: #find direction
                node_src = i
                node_dst = j
            else:
                node_src = j
                node_dst = i
            delta_v = vehicles_empty[node_src] - vehicles_empty[node_dst]
            # if delta_v == 0: #if do have a seat differenc but not a empty vehicle difference
            #     # print(f"Have a seat delta of {delta_s} but do not have any cars on edge ({node_src},{node_dst}).")
            #     # NOTE: this can happen as there can be aspot difference but no vehicle difference and no none-vehicles , but why would not contuineu then?
            #     continue

            #enfore (P2)
            # if delta_s == vehicle_capacity/2: #np.ceil(vehicle_capacity/2):
            #     # do a swap (if can)
            #     # print(f"Have a seat delta of {delta_s} but do not have any cars on edge ({node_src},{node_dst}). Doing a swap.")
            #     spots[node_src] -= vehicle_capacity
            #     spots[node_dst] += vehicle_capacity
            #     vehicles_empty[node_src] -= 1
            #     vehicles_empty[node_dst] += 1
            #     continue

            # just move to smallest delta, enforces (P3)
            # note: this can happen! could also here not even tryh the mvoe if the delta_s < 0.5 vehicle_capacity but my brain is not working so lets keep it
            # move as many vehicles s.t. the difference in spots is minimized (and smaller)
            # implemented as brute force
            move_best = 0
            move_best_value = delta_s
            for move in range(1,delta_v+1):
                move_value = np.abs(spots[node_src]-move*vehicle_capacity - (spots[node_dst]+move*vehicle_capacity))
                if convergence_guarantee:
                    if move_value <= move_best_value:
                        move_best_value = move_value
                        move_best = move
                else:
                    if move_value < move_best_value:
                        move_best_value = move_value
                        move_best = move
            # move by best move: update spots and vehicles
            # print(f"Move {move_best} vehicles from {node_src} to {node_dst}")
            spots[node_src] -= move_best*vehicle_capacity
            spots[node_dst] += move_best*vehicle_capacity
            vehicles_empty[node_src] -= move_best
            vehicles_empty[node_dst] += move_best

    def update_edges_vehicles(vehicles_empty,edges):
        """Update an edge selection aka. move vehciles based on spots values
        
        modifies the arrays spots and vehicles in palce
        
        = perfect balancing"""
        for edge in edges:
            i = edge[0]
            j = edge[1]
            i_new= np.ceil(0.5*(vehicles_empty[i]+vehicles_empty[j]))
            j_new = np.floor(0.5*(vehicles_empty[i]+vehicles_empty[j]))
            vehicles_empty[i] = i_new
            vehicles_empty[j] = j_new
    ### END LOCAL FUNCTIONS

    ### EXPERIMENT LOOP

    # set up storage
    storage_spots = dict() #main storage spots
    storage_vehicles_empty = dict() #main storage vehicles
    for selection_variant in selection_variants:
        for update_variant in update_variants:
            storage_spots[(selection_variant,update_variant)] = np.empty((N_nodes,N_iterations_max,N_experiments), dtype=np.int_)
            storage_vehicles_empty[(selection_variant,update_variant)] = np.empty((N_nodes,N_iterations_max,N_experiments), dtype=np.int_)

    for idx_eperiment in tqdm(range(N_experiments),desc="Experiment"):
        # prepare values
        A,G = generate_stochastic_matrix(n=N_nodes,p=min(1,2/N_nodes),
                                         fully_connected = FULLY_CONNECTED_GRAPH)

        while True:
            # generate initial conditions
            spots_per_vehicle = rng.integers(low=0,high=vehicle_capacity,size=N_vehicles) #amount of free seats = spots per vehicle
            mask = rng.random(spots_per_vehicle.shape) < p_vehicle_empty
            spots_per_vehicle[mask] = vehicle_capacity
            # genrate matching from vehicles to nodes
        
            vehicle_location = rng.choice(a=range(N_nodes),size=N_vehicles,replace=True)
            # fill in actual state vectors
            spots = np.zeros(N_nodes,dtype=np.int_) #amount of spots per node (free and flexible)
            vehicles = np.zeros(N_nodes,dtype=np.int_) # amount of vehicles per node (free and flexible)
            vehicles_empty = np.zeros(N_nodes,dtype=np.int_) #amount of vehicles per node that do not have any people it --> the ones that can actually be moved
            for node in range(N_nodes):
                mask = vehicle_location == node
                vehicles[node] = np.sum(mask)
                vehicles_empty[node] = np.sum(spots_per_vehicle[mask] == vehicle_capacity)
                spots[node] = np.sum(spots_per_vehicle[mask])
            assert np.sum(spots_per_vehicle) == np.sum(spots), "spots got lost"
            assert N_vehicles == np.sum(vehicles), "vehicles got lost"

            # check if the asignment fullfills the assumption
            if not CHECK_ASSUMPTION:
                break

            if np.all(spots - vehicle_capacity*vehicles_empty <= np.ceil(np.mean(spots))):
                #and np.all(spots - vehicle_capacity*vehicles_empty >= np.floor(np.mean(spots))-2*vehicle_capacity):
                break

        # evaluate for "moveable spots" and "unmovable spots"
        if N_experiments == 1:
            fixed_spots = spots - vehicles_empty*vehicle_capacity
            fixed_vehicles = vehicles - vehicles_empty #unmovable vehciles
            print(f"Average of Spots: {np.mean(spots)}")
            print(f"Maximal fixed spot: {np.max(fixed_spots)} ({np.sum(fixed_spots > np.ceil(np.mean(spots)))} nodes have spots higher than the mean spots)")

        # plot graph
        if N_experiments == 1:
            plot_directed_graph(G=G)

        for selection_variant in selection_variants:
            for update_variant in update_variants:
                if N_experiments == 1:
                    print(f"Running selection {selection_variant} with update rule {update_variant}...")

                # set new initial conditions
                spots_changing = spots.copy()
                vehicles_empty_changing = vehicles_empty.copy()

                # Convergence Loop <-- what would run in actual controller
                for interation in tqdm(range(N_iterations_max),desc="Control Loop", leave=False):
                    # print and store iteration
                    # mean_spots = np.mean(spots)
                    # mean_vehicles_empty= np.mean(vehicles_empty)
                    # print(f"[{interation}]\tSum Spots: {np.sum(spots)}\t Sum Empty Vehicles: {np.sum(vehicles_empty)}")
                    # print(f"\tSpots Error:\tL2: {np.linalg.norm(spots-mean_spots,ord=2)}\tL_\inf: {np.linalg.norm(spots-mean_spots,ord=np.inf)}")
                    # print(f"\tEmpty Vehicle Error:\tL2: {np.linalg.norm(vehicles_empty-mean_vehicles_empty,ord=2)}\tL_\inf: {np.linalg.norm(vehicles_empty-mean_vehicles_empty,ord=np.inf)}")
                    storage_spots[(selection_variant,update_variant)][:,interation,idx_eperiment]=spots_changing.copy()
                    storage_vehicles_empty[(selection_variant,update_variant)][:,interation,idx_eperiment]=vehicles_empty_changing.copy()

                    ### (1) Edge Selection ###
                    if selection_variant == 'multiple_uniform':
                        selected_edges = select_edges_uniform(G,vehicles_empty_changing)
                    elif selection_variant == 'single_uniform':
                        selected_edges = select_edges_single_uniform(G,vehicles_empty_changing)
                    elif selection_variant == 'multiple_delta_seat_L1':
                        selected_edges = select_edges_delta_seats(G,vehicles_empty_changing,spots_changing,norm="L1")
                    elif selection_variant == 'multiple_delta_seat_L2':
                        selected_edges = select_edges_delta_seats(G,vehicles_empty_changing,spots_changing,norm="L2")
                    elif selection_variant == 'multiple_delta_vehicle':
                        selected_edges = select_edges_delta_vehicles(G,vehicles_empty_changing)
                    else:
                        raise NotImplementedError(f"Unkown selection variant {selection_variant}.")
                    # print(f"Have {len(selected_edges)} edges selected!")

                    ### (2) Update Edges ###
                    if update_variant == 'spots':
                        update_edges_spots(spots=spots_changing,vehicles_empty=vehicles_empty_changing,edges=selected_edges)
                    elif update_variant == 'spots_convergence':
                        update_edges_spots(spots=spots_changing,vehicles_empty=vehicles_empty_changing,edges=selected_edges,convergence_guarantee=True)
                    elif update_variant == 'vehicles_empty':
                        update_edges_vehicles(vehicles_empty=vehicles_empty_changing,edges=selected_edges)
                    else:
                        raise NotImplementedError(f"Unkown update variant {update_variant}.")
                
                    ### (3) Wait T time, do not do here as already moved ;) ###
    ### END EXPERIMENT LOOP
        
    ### PLOT ###

    def plot_2d_data_spread(ax: Axes, data, color: str,desc: str):
        # prepare data
        min_err = np.min(data, axis=1)         # shape: (n_timepoints,)
        max_err = np.max(data, axis=1)         # shape: (n_timepoints,)
        qlow = np.quantile(data, 0.05, axis=1)  # shape: (n_timepoints,)
        qhigh = np.quantile(data, 0.95, axis=1)
        mean = np.mean(data,axis=1)

        # plot
        ax.fill_between(iterations, qlow, qhigh, alpha=0.1, label='5-95% Quantile', color=color)
        ax.plot(iterations, mean, label=f"Mean {desc}", color=color)
        ax.plot(iterations, min_err, linestyle=':', color=color, alpha=0.6, label='Min/Max')
        ax.plot(iterations, max_err, linestyle=':', color=color, alpha=0.6)


    fig, axes = plt.subplots(len(selection_variants), len(update_variants),figsize=(16,12))
    iterations = range(N_iterations_max)

    for idx_ax_x, selection_variant in enumerate(selection_variants):
        for idx_ax_y, update_variant in enumerate(update_variants):
            ax: Axes = axes[idx_ax_x,idx_ax_y]
            ax_right = ax.twinx()
            # get data
            data_spots =  storage_spots[(selection_variant,update_variant)]
            data_vehicles_empty = storage_vehicles_empty[(selection_variant,update_variant)]

            if N_experiments == 1:
                ax.axhspan(np.floor(np.mean(spots))-vehicle_capacity, np.ceil(np.mean(spots))+vehicle_capacity, facecolor='blue', alpha=0.3, label="Empirical convergence interval")
                for i in range(data_spots.shape[0]):
                    ax.step(iterations, data_spots[i,:,0], where='post',linewidth=1, color='blue') #label=f"Node {i+1} #add this for node labling
                ax.plot(iterations, np.mean(data_spots[:,:,0],axis=0), color='blue', linewidth=2, label='Average Spots', linestyle='--')
                # vehicle data
                # ax_right.axhspan(np.floor(np.sum(vehicles_empty)/N_nodes), np.ceil(np.sum(vehicles_empty)/N_nodes), facecolor='green', alpha=0.3, label="Convergence Interval")
                # for i in range(data_vehicles_empty.shape[0]):
                #     ax_right.step(iterations, data_vehicles_empty[i,:,0], where='post',linewidth=1, color='green', linestyle=':') #label=f"Node {i+1} #add this for node labling
                # ax_right.plot(iterations, np.mean(data_vehicles_empty[:,:,0],axis=0), color='green', linewidth=2, linestyle=':',label='Average Vehicles Empty')
                 # cosmetics
                ax.set_ylabel('Number of Free Seats')
                # ax_right.set_ylabel('Number of Free Vehicles')
                
            else:
                # calculate errors
                error_spots = data_spots - np.mean(data_spots,axis=0)
                l_errors_spots = np.max(np.abs(error_spots),axis=0)
                # l_errors_spots = np.linalg.norm(error_spots,ord=ORDER_ERROR_PLOT,axis=0) #aggregate over nodes
                # if ORDER_ERROR_PLOT == 1 or ORDER_ERROR_PLOT == 2:
                #     l_errors_spots = l_errors_spots / len(error_spots)
                #     additional_text = "(maximal per vehicle)"
                plot_2d_data_spread(ax=ax, data=l_errors_spots, color='blue', desc=f"$\max_i L_{1}$ Error Spot $s_i$")

                # error_vehicles= data_vehicles_empty - np.mean(data_vehicles_empty,axis=0)
                # l_errors_vehicles = np.max(np.abs(error_vehicles),axis=0)
                # # l_errors_vehicles = np.linalg.norm(error_vehicles,ord=ORDER_ERROR_PLOT,axis=0) #aggregate over nodes
                # # if ORDER_ERROR_PLOT == 1 or ORDER_ERROR_PLOT == 2:
                # #     l_errors_vehicles = l_errors_vehicles / len(error_vehicles)
                # #     additional_text = "(per vehicle)"
                # plot_2d_data_spread(ax=ax, data=l_errors_vehicles, color='green', desc=f"$\max_i L_{1}$ Error Vehicles $f_i$")

                # cosmetics
                ax.set_ylabel('Error')
                # ax_right.set_ylabel('Number of Free Vehicles')

            #comon cosmetics
            ax.set_title(f"Selection: `{selection_variant}`, Update: `{update_variant}`")
            ax.set_xlabel("Iterations")
            ax.legend()
            ax_right.legend()
            ax.grid(True)
                

    fig.suptitle("Convergence for different Selection and Update Variants")
    plt.tight_layout()
    plt.show()

    fig.savefig("out/rebalacing_testing.png", dpi=300)
    print("end demo")
 
def paper_experiment_gossip_algorithms():
    """Paper Demo of algorithms
    a stripped variant of the demo"""

    ### SETTINGS
    N_iterations_max = 50 #maximal value
    N_nodes = 50 #NYC has 214
    N_vehicles = 200 #amount of vehilces in total
    vehicle_capacity = 4 #passenger per vehicle
    p_vehicle_empty = 0.2 #probability that a vehicle is empty ~> vehicles that can actually be moved

    assumption_variants = ['no_checking','as_in_lemma'] #['multiple_uniform', 'single_uniform','multiple_delta_seat_L1','multiple_delta_seat_L2','multiple_delta_vehicle']
    update_variants = ['spots_convergence'] #['vehicles_empty','spots','spots_convergence'] 

    N_experiments = 1000 #amount of different experiements i.e. new graph ect.
    ### END SETTINGS

    rng = np.random.default_rng(seed=42)    

    ### LOCAL FUNCTIONS
    def select_edges_uniform(G,vehicles_empty) -> list[tuple[int,int]]:
        """Slect edges in a round uniformly"""
        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # assume uniform probability that the edge is selected; greedily select an edge, then remove all connecting edges of the vertex
        while(len(edges_with_vehicles)>0):
            edge = rng.choice(a=edges_with_vehicles,size=1,replace=False)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            edges_with_vehicles = [
                (i, j) for i, j in edges_with_vehicles
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]

        return selected_edges
    
    def select_edges_single_uniform(G,vehicles_empty) -> list[tuple[int,int]]:
        """Select a single edge (uniformly) from G that has at least one none empty vehicle"""
        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        selected_edges.append(rng.choice(a=edges_with_vehicles,size=1,replace=False)[0])
        return selected_edges
    
    def select_edges_delta_seats(G, vehicles_empty, spots, norm="L2") -> list[tuple[int,int]]:
        """Select edges where edges with a higher spot delta are more likely"""

        selected_edges = []
        edges_with_vehicles = [
            (i,j) for i,j in G.edges()
            if vehicles_empty[i] + vehicles_empty[j] > 0
        ]
        # build probabilities
        probabilities = np.zeros(shape=len(edges_with_vehicles))
        for i, edge in enumerate(edges_with_vehicles):
            if norm == "L2":
                probabilities[i] = max((spots[edge[0]] - spots[edge[1]])**2,0.5) #ensure non-zero probability by counting a spot difference of 0 as 1/2 
            elif norm == "L1":
                probabilities[i] = max(np.abs(spots[edge[0]] - spots[edge[1]]),0.5)
            else:
                raise NotImplementedError(f"Unknown Norm {norm}")
        # draw until cannot draw anymore
        while(len(edges_with_vehicles)>0):
            probabilities = probabilities / np.sum(probabilities, dtype=float) #normalize, need to do at every step as elements are removed
            edge = rng.choice(a=edges_with_vehicles,size=1,replace=False,p=probabilities)[0]
            selected_edges.append(edge)
            # filter out edges witch have this edge
            forbidden_nodes = set(edge)
            index_allowed_edges = [
                index for index, (i,j) in enumerate(edges_with_vehicles)
                if i not in forbidden_nodes and j not in forbidden_nodes
            ]
            probabilities = probabilities[index_allowed_edges]
            edges_with_vehicles = [edges_with_vehicles[i] for i in index_allowed_edges]
        return selected_edges

    def update_edges_spots(spots,vehicles_empty,edges, convergence_guarantee: bool= False):
        """Update an edge selection aka. move vehciles based on spots values
        
        modifies the arrays spots and vehicles in palce"""
        for edge in edges:
            i = edge[0]
            j = edge[1]

            delta_s = np.abs(spots[i] - spots[j])

            # do action
            if delta_s == 0: #equal values # Ensures (P1)
                # print(f"Edge ({i},{j}) is already balanced in terms of seats!")
                continue

            if spots[i] > spots[j]: #find direction
                node_src = i
                node_dst = j
            else:
                node_src = j
                node_dst = i
            delta_v = vehicles_empty[node_src] - vehicles_empty[node_dst]
            # if delta_v == 0: #if do have a seat differenc but not a empty vehicle difference
            #     # print(f"Have a seat delta of {delta_s} but do not have any cars on edge ({node_src},{node_dst}).")
            #     # NOTE: this can happen as there can be aspot difference but no vehicle difference and no none-vehicles , but why would not contuineu then?
            #     continue

            #enfore (P2)
            # if delta_s == vehicle_capacity/2: #np.ceil(vehicle_capacity/2):
            #     # do a swap (if can)
            #     # print(f"Have a seat delta of {delta_s} but do not have any cars on edge ({node_src},{node_dst}). Doing a swap.")
            #     spots[node_src] -= vehicle_capacity
            #     spots[node_dst] += vehicle_capacity
            #     vehicles_empty[node_src] -= 1
            #     vehicles_empty[node_dst] += 1
            #     continue

            # just move to smallest delta, enforces (P3)
            # note: this can happen! could also here not even tryh the mvoe if the delta_s < 0.5 vehicle_capacity but my brain is not working so lets keep it
            # move as many vehicles s.t. the difference in spots is minimized (and smaller)
            # implemented as brute force
            move_best = 0
            move_best_value = delta_s
            for move in range(1,delta_v+1):
                move_value = np.abs(spots[node_src]-move*vehicle_capacity - (spots[node_dst]+move*vehicle_capacity))
                if convergence_guarantee:
                    if move_value <= move_best_value:
                        move_best_value = move_value
                        move_best = move
                else:
                    if move_value < move_best_value:
                        move_best_value = move_value
                        move_best = move
            # move by best move: update spots and vehicles
            # print(f"Move {move_best} vehicles from {node_src} to {node_dst}")
            spots[node_src] -= move_best*vehicle_capacity
            spots[node_dst] += move_best*vehicle_capacity
            vehicles_empty[node_src] -= move_best
            vehicles_empty[node_dst] += move_best

    def update_edges_vehicles(vehicles_empty,edges):
        """Update an edge selection aka. move vehciles based on spots values
        
        modifies the arrays spots and vehicles in palce
        
        = perfect balancing"""
        for edge in edges:
            i = edge[0]
            j = edge[1]
            i_new= np.ceil(0.5*(vehicles_empty[i]+vehicles_empty[j]))
            j_new = np.floor(0.5*(vehicles_empty[i]+vehicles_empty[j]))
            vehicles_empty[i] = i_new
            vehicles_empty[j] = j_new
    ### END LOCAL FUNCTIONS

    ### EXPERIMENT LOOP

    # set up storage
    storage_spots = dict() #main storage spots
    storage_vehicles_empty = dict() #main storage vehicles
    for assumption_variant in assumption_variants:
        for update_variant in update_variants:
            storage_spots[(assumption_variant,update_variant)] = np.empty((N_nodes,N_iterations_max,N_experiments), dtype=np.int_)
            storage_vehicles_empty[(assumption_variant,update_variant)] = np.empty((N_nodes,N_iterations_max,N_experiments), dtype=np.int_)

    for idx_eperiment in tqdm(range(N_experiments),desc="Experiment"):
        for assumption_variant in assumption_variants:
            # prepare values
            A,G = generate_stochastic_matrix(n=N_nodes,p=min(1,2/N_nodes),
                                            fully_connected = assumption_variant != 'no_checking')

            while True:
                # generate initial conditions
                spots_per_vehicle = rng.integers(low=0,high=vehicle_capacity,size=N_vehicles) #amount of free seats = spots per vehicle
                mask = rng.random(spots_per_vehicle.shape) < p_vehicle_empty
                spots_per_vehicle[mask] = vehicle_capacity
                # genrate matching from vehicles to nodes
            
                vehicle_location = rng.choice(a=range(N_nodes),size=N_vehicles,replace=True)
                # fill in actual state vectors
                spots = np.zeros(N_nodes,dtype=np.int_) #amount of spots per node (free and flexible)
                vehicles = np.zeros(N_nodes,dtype=np.int_) # amount of vehicles per node (free and flexible)
                vehicles_empty = np.zeros(N_nodes,dtype=np.int_) #amount of vehicles per node that do not have any people it --> the ones that can actually be moved
                for node in range(N_nodes):
                    mask = vehicle_location == node
                    vehicles[node] = np.sum(mask)
                    vehicles_empty[node] = np.sum(spots_per_vehicle[mask] == vehicle_capacity)
                    spots[node] = np.sum(spots_per_vehicle[mask])
                assert np.sum(spots_per_vehicle) == np.sum(spots), "spots got lost"
                assert N_vehicles == np.sum(vehicles), "vehicles got lost"

                # check if the asignment fullfills the assumption
                if assumption_variant == 'no_checking':
                    break

                if np.all(spots - vehicle_capacity*vehicles_empty <= np.ceil(np.mean(spots))):
                    #and np.all(spots - vehicle_capacity*vehicles_empty >= np.floor(np.mean(spots))-2*vehicle_capacity):
                    break

            # evaluate for "moveable spots" and "unmovable spots"
            if N_experiments == 1:
                fixed_spots = spots - vehicles_empty*vehicle_capacity
                fixed_vehicles = vehicles - vehicles_empty #unmovable vehciles
                print(f"Average of Spots: {np.mean(spots)}")
                print(f"Maximal fixed spot: {np.max(fixed_spots)} ({np.sum(fixed_spots > np.ceil(np.mean(spots)))} nodes have spots higher than the mean spots)")

            # plot graph
            if N_experiments == 1:
                plot_directed_graph(G=G)

            for update_variant in update_variants:
                if N_experiments == 1:
                    print(f"Running selection {assumption_variant} with update rule {update_variant}...")

                # set new initial conditions
                spots_changing = spots.copy()
                vehicles_empty_changing = vehicles_empty.copy()

                # Convergence Loop <-- what would run in actual controller
                for interation in tqdm(range(N_iterations_max),desc="Control Loop", leave=False):
                    # print and store iteration
                    # mean_spots = np.mean(spots)
                    # mean_vehicles_empty= np.mean(vehicles_empty)
                    # print(f"[{interation}]\tSum Spots: {np.sum(spots)}\t Sum Empty Vehicles: {np.sum(vehicles_empty)}")
                    # print(f"\tSpots Error:\tL2: {np.linalg.norm(spots-mean_spots,ord=2)}\tL_\inf: {np.linalg.norm(spots-mean_spots,ord=np.inf)}")
                    # print(f"\tEmpty Vehicle Error:\tL2: {np.linalg.norm(vehicles_empty-mean_vehicles_empty,ord=2)}\tL_\inf: {np.linalg.norm(vehicles_empty-mean_vehicles_empty,ord=np.inf)}")
                    storage_spots[(assumption_variant,update_variant)][:,interation,idx_eperiment]=spots_changing.copy()
                    storage_vehicles_empty[(assumption_variant,update_variant)][:,interation,idx_eperiment]=vehicles_empty_changing.copy()

                    ### (1) Edge Selection ###
                    # if assumption_variant == 'multiple_uniform':
                    #     selected_edges = select_edges_uniform(G,vehicles_empty_changing)
                    # elif assumption_variant == 'single_uniform':
                    #     selected_edges = select_edges_single_uniform(G,vehicles_empty_changing)
                    # elif assumption_variant == 'multiple_delta_seat_L1':
                    #     selected_edges = select_edges_delta_seats(G,vehicles_empty_changing,spots_changing,norm="L1")
                    # elif assumption_variant == 'multiple_delta_seat_L2':
                    selected_edges = select_edges_delta_seats(G,vehicles_empty_changing,spots_changing,norm="L2")
                    # elif assumption_variant == 'multiple_delta_vehicle':
                    #     selected_edges = select_edges_delta_vehicles(G,vehicles_empty_changing)
                    # else:
                    #     raise NotImplementedError(f"Unkown selection variant {assumption_variant}.")
                    # print(f"Have {len(selected_edges)} edges selected!")

                    ### (2) Update Edges ###
                    if update_variant == 'spots':
                        update_edges_spots(spots=spots_changing,vehicles_empty=vehicles_empty_changing,edges=selected_edges)
                    elif update_variant == 'spots_convergence':
                        update_edges_spots(spots=spots_changing,vehicles_empty=vehicles_empty_changing,edges=selected_edges,convergence_guarantee=True)
                    elif update_variant == 'vehicles_empty':
                        update_edges_vehicles(vehicles_empty=vehicles_empty_changing,edges=selected_edges)
                    else:
                        raise NotImplementedError(f"Unkown update variant {update_variant}.")
                
                    ### (3) Wait T time, do not do here as already moved ;) ###
    ### END EXPERIMENT LOOP
        
    ### PLOT ###
    plt.style.use(['science','ieee'])
    alpha = 2
    figsize_x = 3.5*alpha
    figsize_y = 1*alpha
    # plt.rcParams.update({
    #     "font.size": 4,        # base font size
    #     "axes.labelsize": 4,   # axis labels
    #     "xtick.labelsize": 3,
    #     "ytick.labelsize": 3,
    #     "legend.fontsize": 3
    # })
    def plot_2d_data_spread(ax: Axes, data, color: str,desc: str):
        # prepare data
        min_err = np.min(data, axis=1)         # shape: (n_timepoints,)
        max_err = np.max(data, axis=1)         # shape: (n_timepoints,)
        qlow = np.quantile(data, 0.05, axis=1)  # shape: (n_timepoints,)
        qhigh = np.quantile(data, 0.95, axis=1)
        mean = np.mean(data,axis=1)

        # plot
        ax.fill_between(iterations, qlow, qhigh, alpha=0.1, label=f"5\%-95\% Quantile {desc}", color=color)
        ax.plot(iterations, mean, label=f"Mean {desc}", color=color)
        # ax.plot(iterations, min_err, linestyle=':', color=color, alpha=0.6, label='Min/Max')
        # ax.plot(iterations, max_err, linestyle=':', color=color, alpha=0.6)

    assert len(update_variants) == 1, "This is just for the paper!"
    fig, axes = plt.subplots(1,len(assumption_variants),figsize=(figsize_x, figsize_y), sharey=True)
    iterations = range(N_iterations_max)

    for idx_ax_x, assumption_variant in enumerate(assumption_variants):
        for idx_ax_y, update_variant in enumerate(update_variants):
            ax: Axes = axes[idx_ax_x]
            # get data
            data_spots =  storage_spots[(assumption_variant,update_variant)]

            # calculate errors
            error_spots = data_spots - np.mean(data_spots,axis=0)
            l_errors_spots_max = np.max(np.abs(error_spots),axis=0)
            l_errors_wrongorder= np.abs(error_spots)
            arr_reordered = np.transpose(l_errors_wrongorder, (1, 0, 2))
            l1_errors_spots = arr_reordered.reshape(N_iterations_max,N_nodes*N_experiments)
            # l_errors_spots = np.linalg.norm(error_spots,ord=ORDER_ERROR_PLOT,axis=0) #aggregate over nodes
            # if ORDER_ERROR_PLOT == 1 or ORDER_ERROR_PLOT == 2:
            #     l_errors_spots = l_errors_spots / len(error_spots)
            #     additional_text = "(maximal per vehicle)"
            ax.axhspan(0, 1+vehicle_capacity, facecolor='gray', alpha=0.2, label="Convergence Interval")
            plot_2d_data_spread(ax=ax, data=l1_errors_spots, color='blue', desc="$L_{1}$ Error")
            # plot_2d_data_spread(ax=ax, data=l_errors_spots_max, color='blue', desc=f"$\max_i L_{1}$ Error Spot $s_i$")
            ax.plot(iterations, np.max(l_errors_spots_max,axis=1), linestyle='-', color='red', label="$\max_{\\text{Experiments},\\text{Regions}} L_{1}$ Error")

            # error_vehicles= data_vehicles_empty - np.mean(data_vehicles_empty,axis=0)
            # l_errors_vehicles = np.max(np.abs(error_vehicles),axis=0)
            # # l_errors_vehicles = np.linalg.norm(error_vehicles,ord=ORDER_ERROR_PLOT,axis=0) #aggregate over nodes
            # # if ORDER_ERROR_PLOT == 1 or ORDER_ERROR_PLOT == 2:
            # #     l_errors_vehicles = l_errors_vehicles / len(error_vehicles)
            # #     additional_text = "(per vehicle)"
            # plot_2d_data_spread(ax=ax, data=l_errors_vehicles, color='green', desc=f"$\max_i L_{1}$ Error Vehicles $f_i$")

            # cosmetics
            ax.set_ylabel('Error')
            # ax_right.set_ylabel('Number of Free Vehicles')

            #comon cosmetics
            ax.set_xlabel("Iteration")
            ax.grid(True)
    axes[0].set_title("General Graph $G_R$")
    axes[1].set_title("Assumptions as in Lemma 2")
    axes[1].legend()
                

    # fig.suptitle("Numerical Convergence of Static Rebalancing Algorithm")
    plt.tight_layout()
    # plt.show(block=False)

    fig.savefig("out/rebalacing_static_paper.png", dpi=300)
    fig.savefig("out/rebalacing_static_paper.svg", dpi=300)
    print("end paper demo")

 ### RUN ###
if __name__ == "__main__":
    print("BEGIN OF REBALANCING PLAYGROUND")

    # demo_perfect_balancing()
    # demo_gossip_algorithms()
    paper_experiment_gossip_algorithms()

    print("END OF REBALANCING PLAYGROUND")
