# -*- coding: utf-8 -*-
"""
Created on jun 2020
v 0.3

@author: bobobert

All new rollout sampler function using the cuda compiler from the libraty Numba.
In here you will find the necessary to generate to pass an actual state of an
Helicopter environment to create the trajectories to sample, the kernel to run them
and the function to pass the objective function back.
WARNING!
The heuristic can't be passed yet as an object, they need to be reprogramed in here 
with the cuda.jit(device=True) decorator to compile and run in a thread.
"""
# TODO 
# add prunning to the sampler inside the n_sample bucle


# Rob was here
from numba import cuda
import numba as nb
import numpy as np
import math
import itertools
# Numba random methods. 
from numba.cuda.random import create_xoroshiro128p_states as rdm_states_gen
from numba.cuda.random import xoroshiro128p_uniform_float32 as rdm_uniform_sample
# min_max generates some deprecations messages, this will turn them off for looks.
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# These values needs to be here an declared for the compiler. One can change
# then before to create any kernel

THREADSPREAD = 64

NPTFLOAT = np.float32
NPTINT = np.int16
NBTFLOAT = nb.float32
NBTINT = nb.int16

EMPTY = 0
TREE = 1
FIRE = 7

N_SAMPLES = 30
LOOKAHEAD = 7
K = 50
# This are to make the movements, change them
# before calling the kernel if necessary
ACTION_SET = [1,2,3,4,5,6,7,8,9]
L_AS = 9

#            n_row, n_col
GRID_SIZE = (   16,   16)

# Design sizes, do not change until the kernel functions are changed as well
PARAMETERS_SIZE = 7
COSTS_SIZE = 6
PROBS_SIZE = 2


###### PROGRAM HERE YOUR HEURISTICs ######
### ASING THEM INTO A SWITH MODE USE #####
@cuda.jit(device=True)
def Heuristic(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker):
    h_mode = parameters[6]

    if h_mode == 0:
        # dummy heuristic, by rob. It took around 5 hrs to write this one
        return 3

    elif h_mode == 11:
        # Heuristic from Mau
        # Corrective, vision = 1
        action = 5
        VISION = 1
        burned = 0
        # Count the burned cells on the neighborhood
        for i in range(pos_row - VISION, pos_col + VISION + 1):
            for j in range(pos_col - VISION, pos_col + VISION + 1):
                if local_grid[i,j] == FIRE:
                    burned += 1
        if burned == 0:
            for i in range(1, 10):
                if rdm_uniform_sample(random_states, worker) < 0.112:
                    action = i
        else:
            p =  1 / burned
            for i in range(pos_row - VISION, pos_col + VISION + 1):
                for j in range(pos_col - VISION, pos_col + VISION + 1):
                    if (i < 0) or (i >= GRID_SIZE[0]):
                        # Out of boundaries, can't go there.
                        0.0
                    elif (j < 0) or (j >= GRID_SIZE[1]):
                        # Out of boundaries, can't go there
                        0.0
                    elif local_grid[i,j] == FIRE:
                        # This could be the action
                        if action == 0:
                            # Go greedy
                            action = 3*(i - pos_row + 1) + (j - pos_col + 1) + 1
                        elif rdm_uniform_sample(random_states, worker) < p:
                            # Take a coin to chose it or not
                            action = 3*(i - pos_row + 1) + (j - pos_col + 1) + 1
        return action

    elif h_mode == 12:
        return CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 2)
    elif h_mode == 13:
        return CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 3)
    elif h_mode == 21:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 1)
    elif h_mode == 22:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 2)
    elif h_mode == 23:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 3)

@cuda.jit(device=True)
def CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, VISION):
    # Heuristic from Mau
    # Corrective, vision = 2
    action = 5
    NS = 3 + 2 * VISION
    cb = cuda.local.array(9, dtype=NBTINT)
    for i in range(9):
        cb[i] = 0
    # top_left top  top_right  left  --  rigth down_left  down  down_right
    #    0      1       2        3    4    5       6       7        8
    # Count the burned cells on the neighborhood
    for i in range(pos_row - VISION, pos_col + VISION + 1):
        for j in range(pos_col - VISION, pos_col + VISION + 1):
            if (i < 0) and (i >= GRID_SIZE[0]):
                0.0
            elif (j < 0) and (j >= GRID_SIZE[1]):
                0.0 # Out of boundaries
            elif local_grid[i,j] == FIRE:
                if (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < NS):
                    #Up zone
                    cb[1] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < 1 + VISION):
                    #Up left zone
                    cb[0] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j > 1 + VISION) and (j < NS):
                    #Up rigth zone
                    cb[2] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0 ) and (j < NS):
                    # Down zone
                    cb[7] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Down left zone
                    cb[6] += 1
                elif (i > 1 + VISION) and (i < NS) and (j > 1 + VISION) and (j < NS):
                    #Down right zone
                    cb[8] += 1
                elif (i >= 0) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Left zone
                    cb[3] += 1
                elif (i >= 0) and (i < NS) and (j > 1 + VISION ) and (j < NS):
                    #Right zone
                    cb[5] += 1

    non_zero = 0
    for i in range(9):
        if cb[i] > 0:
            non_zero += 1
    if non_zero == 0:
        for i in range(1, 10):
            if rdm_uniform_sample(random_states, worker) < 0.112:
                action = i
    else:
        p = 1 / non_zero
        max_yeet = 0        
        for i in range(9):
            if cb[i] > max_yeet:
                action = i + 1
                max_yeet = cb[i]
            elif cb[i] == max_yeet:
                if rdm_uniform_sample(random_states, worker) < p:
                    action = i + 1
    return action

@cuda.jit(device=True)
def PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, VISION):
    # Heuristic preventive variable vision
    # based on Mau's
    action = 5
    VISION = 1
    NS = 3 + 2 * VISION
    cb = cuda.local.array(9, dtype=NBTFLOAT)
    cz = cuda.local.array(9, dtype=NBTINT)
    for i in range(9):
        cb[i] = 0.0
        cz[i] = 0
    # Function coefficients
    Fire_coef = 2.0
    Tree_coef = 0.5
    Empty_coef = 0.5
    # top_left top  top_right  left  --  rigth down_left  down  down_right
    #    0      1       2        3    4    5       6       7        8
    # Count the burned cells on the neighborhood
    for i in range(pos_row - VISION, pos_col + VISION + 1):
        for j in range(pos_col - VISION, pos_col + VISION + 1):
            if (i < 0) and (i >= GRID_SIZE[0]):
                0.0
            elif (j < 0) and (j >= GRID_SIZE[1]):
                0.0 # Out of boundaries
            else:
                if (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < NS):
                    #Up zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[1] += coef
                    cz[1] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < 1 + VISION):
                    #Up left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[0] += coef
                    cz[0] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j > 1 + VISION) and (j < NS):
                    #Up rigth zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[2] += coef
                    cz[2] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0 ) and (j < NS):
                    # Down zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[7] += coef
                    cz[7] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Down left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[6] += coef
                    cz[6] += 1
                elif (i > 1 + VISION) and (i < NS) and (j > 1 + VISION) and (j < NS):
                    #Down right zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[8] += coef
                    cz[8] += 1
                elif (i >= 0) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[3] += coef
                    cz[3] += 1
                elif (i >= 0) and (i < NS) and (j > 1 + VISION ) and (j < NS):
                    #Right zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[5] += coef
                    cz[5] += 1

    for i in range(9):
        # Normalize the coefficients
        if cz[i] != 0:
            cb[i] = cb[i] / cz[i]
        else:
            cb[i] = 0

    non_zero = 0
    for i in range(9):
        if cb[i] > 0:
            non_zero += 1
    if non_zero == 0:
        for i in range(1, 10):
            if rdm_uniform_sample(random_states, worker) < 0.112:
                action = i
    else:
        p = 1 / non_zero
        max_yeet = 0.0
        for i in range(9):
            if cb[i] > max_yeet:
                action = i + 1
                max_yeet = cb[i]
            elif cb[i] == max_yeet:
                if rdm_uniform_sample(random_states, worker) < p:
                    action = i + 1
    return action
############## END OF IT ################


def load_forest_fire_2_device(grid=None, grid_size=(20,20),
                            ip_tree=0.5, p_fire=0.001, p_tree=0.005,
                            alpha = 0.99, k = 50,
                            reward_tree = -3.0, reward_fire = 1.5, reward_empty = 0.0, 
                            reward_hit = -0.20, reward_move = 1.0,
                            pos=(7,7),
                            actual_steps_before_update = 4,
                            steps_before_update = 4,
                            seed=None, h_mode=0):
    """
    This is a function that returns an explicit list to the device memory for it to access
    it for the other functions running on the device threads. Run before the kernel call to load
    the initial state.
    Parameters
    ----------
    grid: numpy array or tuple int 
        Is None provided, it generates one random grid with EMPTY(0), TREE(1) cell type. Fire can 
        be on the grid after the next iterations, this cell type is (7).
    grid_size: tuple of ints
        If grid=None, this tuple must be given to generate a grid to the size to match.
    ip_tree: float
        Probability for trees to spawn in the initial generation of the grid. As the distribution
        is uniform, this probability is in average the ratio of the trees on the grid.
    p_fire: float
        Probability for a lightning to hit a cell.
    p_tree: float
        Probability for a tree to grow from an empty cell type.
    pos: tuple of int
        Position in (row, col) for the helicopter agent.
    actual_steps_before_update: int
        The number of steps left on the actual position of the agent for the environment
        to update its grid.
    steps_before_update: int
        The steps the agent can take before the environment updates.
    h_mode: int
        An integer to call
    """
    def check_probs(probs):
        probs = np.array(probs, dtype=NPTFLOAT)
        s_probs = np.sum(probs)
        if s_probs > 1:
            # Normalizing if the probability is larger
            probs = probs / s_probs
        return probs
    # Grid loading/generating
    if grid is None:
        # Generating grid, no Fire
        grid = np.random.choice([TREE, EMPTY], size=grid_size, p=[ip_tree, 1 - ip_tree])
    else:
        # Saving the a copy of the grid
        grid = grid.copy()
    # Updating the probabilities passed
    #                      p_fire, p_tree
    #                        0      1
    probs = check_probs([p_fire, p_tree])
    if seed == None:
        seed = int(np.random.random()*10**3)
    # This will be a int type array
    #            pos_row, pos_col, actual_steps_before_update, steps_before_update, k, seed, h_mode
    #               0        1              2                          3            4   5       6
    parameters = [pos[0], pos[1], actual_steps_before_update, steps_before_update, k, seed, h_mode]
    parameters = np.array(parameters, dtype=NPTINT)
    #        costs_tree,  cost_fire,   cost_empty,   cost_hit,   cost_move,   alpha
    #             0           1            2             3           4          5
    costs = [reward_tree, reward_fire, reward_empty, reward_hit, reward_move, alpha]
    costs = np.array(costs, dtype=NPTFLOAT)
    #print(f_probs, parameters, costs)
    # Load to device
    grid_mem = cuda.to_device(np.array(grid, dtype=NPTINT))
    probs_mem = cuda.to_device(probs)
    params_mem = cuda.to_device(parameters)
    costs_mem = cuda.to_device(costs)
    return grid_mem, probs_mem, params_mem, costs_mem # Pass references on device.

def get_leafs(action_set,
              depth, 
              n_samples):
    """
    Function that generates a list for the kernel to asign samples trajectories to the
    device. It returns in a numpy array all the leafs of the tree with the lookahead depht
    required.

    Parameters
    ----------
    actions_set: list
        All the actions possible to iterate from. Accepts other iterables.
    depth: int
        The lookahead wanted of the sample tree
    n_samples: int
        Number of samples required per leaf of the three.
    """
    leafs = [trajectory for trajectory in itertools.product(action_set,repeat=depth)]
    return np.array(leafs, dtype=np.int8)
    # Deprecated.
    """if n_samples >= 1:
        leafs = [trajectories for trajectories in itertools.repeat(leafs, n_samples)]
        leafs = np.array(leafs, dtype=NPTINT)
        return leafs.reshape(leafs.shape[0]*leafs.shape[1], leafs.shape[2])
    else:
        raise Exception("n_samples need to be equal or greater than 1. {} was given.".format(n_samples))"""

@cuda.jit
def sample_trajectories(grid, 
                        probs, 
                        parameters, 
                        costs, 
                        trajectories, 
                        random_states, 
                        results):
    """
    New function to sample all the trajectories individually from all the posible
    trayctories and the samples; for each there's a repeated trajectory. This is made this way
    to use the most of the device resources most of the time to accelerate the process.
    Very ad-hoc for the forest fire environment
    """
    # Loading in shared memory everything needed to sample.
    env_initial_state = cuda.shared.array(GRID_SIZE, dtype=nb.int8)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            env_initial_state[i,j] = grid[i,j]

    # Starting linear addressing of the samples
    worker = cuda.grid(1)

    if worker < trajectories.shape[0]:
        # A local copy of the grid
        local_grid = cuda.local.array(GRID_SIZE, dtype=nb.int8)
        # Updated works to generate a copy from here to better behavior
        updated_grid = cuda.local.array(GRID_SIZE, dtype=nb.int8)
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                local_grid[i,j] = env_initial_state[i,j]
                updated_grid[i,j] = EMPTY
        # Doing a sample given a trajectory
        # Initial conditions
        pos_row, pos_col = parameters[0], parameters[1]
        steps_to_update = parameters[2]
        ALPHA = costs[5] # Loading alpha from shared memory
        alpha = ALPHA
        sample_cost = 0.0
        # Begin the sample
        for action in trajectories[worker]:
            updated_grid, pos_row, pos_col, steps_to_update, cost = \
                helicopter_step(local_grid, updated_grid, \
                                pos_row, pos_col, steps_to_update, action, \
                                parameters, probs, costs, random_states, worker)
            sample_cost += cost*alpha #Save cost
            alpha *= ALPHA # Updating alpha
            # Updating local_grid
            for i in range(GRID_SIZE[0]):
                for j in range(GRID_SIZE[1]):
                    local_grid[i,j] = updated_grid[i,j]
        # End of trajectory
        # Begining to use the heuristic
        k = parameters[4]
        for _ in range(k):
            action = Heuristic(local_grid, pos_row, pos_col, steps_to_update, \
                                parameters, probs, costs, random_states, worker)
            updated_grid, pos_row, pos_col, steps_to_update, cost = \
                helicopter_step(local_grid, updated_grid, \
                                pos_row, pos_col, steps_to_update, action, \
                                parameters, probs, costs, random_states, worker)
            sample_cost += cost*alpha #Save cost
            alpha *= ALPHA # Updating alpha
            # Updating local_grid
            for i in range(GRID_SIZE[0]):
                for j in range(GRID_SIZE[1]):
                    local_grid[i,j] = updated_grid[i,j]
        results[worker] = sample_cost # Finishing worker, saving result into results memory

@cuda.jit(device=True)
def helicopter_step(grid, 
                    updated_grid, 
                    pos_row, pos_col, 
                    steps_before_update, 
                    action, 
                    parameters, 
                    probs, 
                    costs, 
                    random_states, 
                    worker):
    """
    Function on to execute on device per thread to update
    and give a step of the agent in its environment. This is meant
    to run inside the sample_trajectories kernel.
    """

    new_steps_before_updating = 0

    # Check if the grid needs to be updated
    if steps_before_update == 0:
        # Generating the random values
        throws = cuda.local.array(GRID_SIZE, dtype=NBTFLOAT)
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                throws[i,j] = rdm_uniform_sample(random_states, worker)

        # Begin the update of the grid.
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                # From the cell padded_grid[i,j]
                # If it's a fire, spread on its neighborhood
                if grid[i,j] == FIRE:
                    # Get the neighborhood
                    for i_n in range(i-1, i+2):
                        for j_n in range(j-1, j+2):
                            if (i_n < 0) or (i_n >= GRID_SIZE[0]):
                                0.0 #Out of bounds
                            elif (j_n < 0) or (j_n >= GRID_SIZE[1]):
                                0.0 # Out of bounds
                            elif grid[i_n,j_n] == TREE:
                                # Burn the tree
                                updated_grid[i_n,j_n] = FIRE
                    # Extinguish the fire
                    updated_grid[i,j] = EMPTY
                # If it's a tree, throw a dice to a lighting to
                # hit it or not
                elif (grid[i,j] == TREE) and (throws[i,j] <= probs[0]):
                        # The tree is hitted by a ligthning
                        updated_grid[i,j] = FIRE
                # If the cell it's empty, it has a chance to grow a tree
                elif (grid[i,j] == EMPTY) and (throws[i,j] <= probs[1]):
                        # A tree growns in this cell
                        updated_grid[i,j] = TREE
        new_steps_before_updating = parameters[3] # Restarting

    else:
        new_steps_before_updating = steps_before_update - 1
    # End of the grid update

    # Start of the agent movements
    # By design, if the grid changes happens at the same time that the
    # grids updates. So it it was to put down a fire, this will dissapear
    # and wont count.
    it_moved = 1.0
    delta_row, delta_col = 0, 0
    if action == 5:
        0.0
    elif (action == 1) or (action == 2) or (action == 3):
        delta_row = -1
    elif (action == 7) or (action == 8) or (action == 9):
        delta_row = 1
    if (action == 1) or (action == 4) or (action == 7):
        delta_col = -1
    elif (action == 3) or (action == 6) or (action == 9):
        delta_col = 1

    new_pos_row = pos_row + delta_row
    new_pos_col = pos_col + delta_col

    if (new_pos_row < 0) or (new_pos_row >= GRID_SIZE[0]):
        # Invalid movement - out of bounds
        new_pos_row = pos_row
    if (new_pos_col < 0) or (new_pos_col >= GRID_SIZE[1]):
        # Invalid movement - our of bounds
        new_pos_col = pos_col
    if (pos_row == new_pos_row) and (pos_col == new_pos_col):
        # There was no movement, the action was out of bounds
        it_moved = 0.0
    # End of agent movement

    # Start to check the hits
    hits = 0.
    if updated_grid[new_pos_row, new_pos_col] == FIRE:
        updated_grid[new_pos_row, new_pos_col] = EMPTY
        hits += 1.0
    # End of hits

    # Start to counting the cells
    fires, empties, trees = 0.0 ,0.0 ,0.0
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if updated_grid[i,j] == FIRE:
                fires += 1.
            elif updated_grid[i,j] == TREE:
                trees += 1. 
            elif updated_grid[i,j] == EMPTY:
                empties += 1.
    # End of counting

    # Calculating cost. 
    ### This is the cost shape for a given state ### 
    ### This is the same as 'custom' type on Helicopter Env ###
    cost = 0.0
    cost += costs[0]*trees
    cost += costs[1]*fires
    cost += costs[2]*empties
    cost += costs[3]*hits
    cost += costs[4]*it_moved
    # End of cost

    return updated_grid, new_pos_row, new_pos_col, new_steps_before_updating, cost

@nb.jit
def min_max(trajectories, results, n_samples, action_set, min_obj):
    """
    A custon function to calculate the means of the trajectories
    given only the first action of the trajectory. Then calculates
    the minimum.
    """
    l_as = len(action_set)
    sample_avg = np.zeros(l_as, dtype=NPTFLOAT)
    sample_c = np.ones(l_as, dtype=np.uint32)
    for k in range(n_samples):
        # Calculating the averages for the first action in the trajectory
        for t, c in zip(trajectories, results[k]):
            fa = t[0] # Getting the first action
            for i in range(l_as):
                if fa == action_set[i]:
                    #Store the 
                    s_c = sample_c[i]
                    sample_avg[i] = sample_avg[i]*(s_c - 1) / s_c + c / s_c
                    sample_c[i] += 1

    best_action, best_cost, obj = 0, np.inf, 1

    if not min_obj:
        # Maximize
        obj = -1
        best_cost *= obj
  
    for i in range(l_as):
        c = sample_avg[i]
        if obj*c < obj*best_cost:
            best_cost = c
            best_action = action_set[i]
        elif obj*c == obj*best_cost:
            if np.random.random() < 0.112:
                best_action = action_set[i]

    action_avg = np.zeros((l_as,2), dtype=NPTFLOAT)
    for i in range(l_as):
        action_avg[i,0] = sample_avg[i]
        action_avg[i,1] = action_set[i]
    return best_action, best_cost, action_avg

def sampler(env,
            h_mode=0,
            alpha=1.0,
            n_samples = 30,
            k=50,
            lookahead=1,
            min_obj=True,
            action_set=None,
            seed=1):
    """
    Function to pass an Helicopter environment and the other parameters
    to sample all the actions from the generated tree given the lookahead
    on a GPU CUDA Capable device. 

    Make sure your PC has the adecuate requeriments on CUDATOOLKIT and 
    Numba. 

    At the moment the GPU kenerl is compiled with each call of this function.
    Parameters
    ----------
    env: class Type Helicopter
        The environment which actual state will the initial state for all
        samples
    h_mode: int
        From the heurititcs programmed in this package. Pass the mode one
        wants the heuristic to run during all the samples.
    alpha: float
        Disconunt value that will be applied to each step cost. This is done
        in a manner that the farther costs can have a less impact on the sample
        cost overall.
    n_samples: int
        Quantity of samples to execute per trajectory on the tree.
    k: int
        Steps to execute with the heuristic given. It can be zero if no heuristic
        is required
    lookahead: int
        The depth of the trajectory tree.
    min_obj: bool
        If the objective is to minimize set to true, otherwise False.
    action_set: list
        If one wants to change the action_set from the package give a list. Otherwise
        leave it in None.
    """
    #Checking for cuda devices
    is_device = False
    for device in cuda.gpus:
        is_device = True
    assert is_device, \
        "No Cuda capable gpu found. Please check your numba and cudatoolkit if you have a cuda capable deivce."
    assert (env.boundary == 'invariant') and \
        (env.forest_mode == 'stochastic'), \
        "The actual environment configuration is not supported in this sampler"
    # CHANGING GLOBALS VARIABLES
    SAMPLER_CONST = globals()
    SAMPLER_CONST['LOOKAHEAD'] = lookahead
    SAMPLER_CONST['N_SAMPLES'] = n_samples
    SAMPLER_CONST['K'] = k
    SAMPLER_CONST['GRID_SIZE'] = env.grid.shape
    SAMPLER_CONST['EMPTY'] = env.empty
    SAMPLER_CONST['TREE'] = env.tree
    SAMPLER_CONST['FIRE'] = env.fire
    if not action_set is None:
        SAMPLER_CONST['ACTION_SET'] = action_set
        SAMPLER_CONST['L_AS'] =  len(action_set)
    # Generating the trajectories from the tree leafs
    trajectories = get_leafs(ACTION_SET, LOOKAHEAD, 1)
    # Copying to device
    d_trajectories = cuda.to_device(trajectories)
    d_results = cuda.device_array(trajectories.shape[0],dtype=NPTFLOAT)
    # Calculating kernel size
    threadspread = THREADSPREAD # Pascal has SM with 64 or 128 units
    blockspread = math.ceil(trajectories.shape[0] / threadspread)
    # Setting Random generators
    random_states = rdm_states_gen(threadspread*blockspread, seed=seed)
    #Loading the actual state of the environment to the device
    d_grid, d_probs, d_params, d_costs = load_forest_fire_2_device(
        grid=env.grid,
        p_fire=env.p_fire,
        p_tree= env.p_tree,
        alpha=alpha,
        reward_tree=env.reward_tree,
        reward_fire=env.reward_fire,
        reward_empty=env.reward_empty,
        reward_hit=env.reward_hit,
        reward_move=env.reward_move,
        pos=(env.pos_row, env.pos_col),
        actual_steps_before_update=env.remaining_moves,
        steps_before_update=env.moves_before_updating,
        h_mode=h_mode,
        k=k
    )
    # Starting the kernel on the device to sample the trajectories
    results = []
    for _ in range(n_samples):
        sample_trajectories[blockspread, threadspread](d_grid, \
            d_probs, d_params, d_costs, d_trajectories, random_states, d_results)
        # Retriving to host the samples results                        (grid, probs, parameters, costs, trajectories, random_states, results)
        results.append(d_results.copy_to_host())
    # Applying objective
    results = np.array(results)
    best_action, best_cost, avg_costs_action = min_max(trajectories, results, N_SAMPLES, ACTION_SET, min_obj)

    # Returning to original values to action set only.
    if not action_set is None:
        SAMPLER_CONST['ACTION_SET'] = [1,2,3,4,5,6,7,8,9]
        SAMPLER_CONST['L_AS'] =  9

    return best_action, best_cost, avg_costs_action