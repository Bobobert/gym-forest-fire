# -*- coding: utf-8 -*-
"""
Created on jun 2020

@author: bobobert
"""
# Rob was here

import math

class R_ITER():
    """Class for to save a bit of space in creation of iterables with repeated items
    for a iterable object where just the action variable changes, iterating from the
    Action_set.

    Parameters
    ----------
    env: Class or object to environment
        Environment must have a .copy() method
    H : ref to class or function for the heuristic
    N_workers : int
        Expected number of workers to have, this is needed here to make all the trayectories
        needed to sample somehow distributed uniformly as possible among the workers.
    alpha: float
    K : int
    lookahead : int
    N_samples : int
    Action_set : list or set
        It can be none if the environment provides it with as env.action_set
    H_args : dict
        Extra arguments to pass the Heuristic
    Min_obj : bool
        If one wants the sampler to compare the costs with minimization or maximization
    """
    def __init__(self, env, H, N_workers, alpha, K, lookahead, 
                    N_samples, Action_set=None, H_args=None, Min_obj=True):
            self.env = env
            self.H_ref = H
            self.Min_obj = Min_obj
            self.cpus = N_workers
            self.alpha = alpha
            if Action_set is None:
                Action_set = env.action_set
            assert isinstance(Action_set, (list, set)), "Action_set variable must be a list or set. {} was given".format(type(Action_set))
            self.actions = list(Action_set)
            self.actions_c = len(self.actions)
            self.K = K
            self.lookahead = lookahead
            self.N_samples = N_samples
            self.H_args = H_args
            self.chunks = math.ceil(self.actions_c**lookahead / self.cpus) # Size of chuncks
            self.j = lookahead - 1
            self.max_index = []
            self.t_pos = []
            for _ in range(self.lookahead):
                self.t_pos += [0]
                self.max_index += [self.actions_c - 1]
            self.t_pos[self.lookahead - 1] = -1 # A quick fix.
            self.all_trayectories = [] # Storing all trayectories. It's better to complextity to memory than
                # to process it everytime
            self.done = self._construct_
            self.i = 0  
    @property
    def _construct_(self):
        self.is_constructed = False
        while not self.is_constructed:
            chunck_trayectories = []
            for _ in range(self.chunks):
                branch = []
                self._update_iter_
                if self.is_constructed:
                    break
                else:
                    for pos in self.t_pos:
                        branch += [self.actions[pos]]
                    chunck_trayectories += [branch]
            self.all_trayectories += [chunck_trayectories]
        return True
    @property
    def _update_iter_(self):
        # A recursive counter
        if self.j < 0:
            # Done
            self.is_constructed = True
        elif self.t_pos[self.j] < self.max_index[self.j]:
            # This case is when there are still branches to 
            # visit in this level
            self.t_pos[self.j] += 1
            self.j = self.lookahead - 1
        else:
            self.t_pos[self.j] = 0
            self.j -= 1
            self._update_iter_
    def __iter__(self):
        # iterable method
        if not self.done:
            self.done = self._construct_
        self.i = 0
        return self
    def __next__(self):
        if self.i == self.cpus:
            raise StopIteration
        else:
            To_H = {
                    'env': self.env.copy(), # Always a new copy of the environment. This is a new object.
                    'H':self.H_ref,
                    'trayectories':self.all_trayectories[self.i],
                    'alpha':self.alpha,
                    'K':self.K,
                    'N_SAMPLES':self.N_samples,
                    'H_args':self.H_args,
                    'min_obj':self.Min_obj,
            }
            self.i += 1
            return To_H
    def __del__(self):
        return None

def sample_trayectory(args):
    """
    This function is the sampler for the rollout function design to 
    run in one thread for parallel calling. It supports a list of 
    trayectories. At the end returns the best action with the expected
    cost associated.

    It returns the expect value from taking the action.
    (action type, float)

    args - dict
    -------------
    'env' : Variable to reference the environmet to work with.
        It is expect to be a COPY from the
        object class Helicopter for this version.
    'trayectories' : list
        An order list with the actions to perfomr to make a trayectory of controls
        to sample.
    'H' : Heuristic to perform the next K steps after executing the action.
    'H_args'* : Optional argument dict for H.
    'alpha' : Discount factor for the cost function.
    'K', Integer of number of steps to keep executing the heuristic.
    'N_SAMPLES' : Number of samples required to calculate the expected value of the
        cost function.
    'min_obj' : bool
        The objective of the samples. To minimize or to maximize if false.
    """
    LIMIT_OF_STEPS = 10**3
    # loading variables
    expected_args = ['env', 'trayectories', 'H', 'alpha', 'K', 'N_SAMPLES','min_obj']
    assert isinstance(args, dict), "This function requieres a dict of kwards to perform. Please provide one"
    for arg in expected_args:
        if args.get(arg) is None:
            raise "At least the argument {} is missing! Please do pass correct arguments".format(arg)
    env = args['env']

    trayectories = args['trayectories']
    l_trayectories = len(trayectories)
    assert l_trayectories > 0, "This function should not sample empty trayectories. Please check your inputs."
    l_trayectory = len(trayectories[0])

    H = args['H']
    H_args = args['H_args'] #Dictionary of aditional args to the H
    To_H = dict()
    To_H['env'] = env # Dict to pass the Heuristic
    if isinstance(H_args, dict): # There are extra arguments to pass
        for extra in H_args.keys():
            To_H[extra] = H_args[extra]

    alpha = args['alpha']
    
    K = args['K']
    if K < 0: 
        K = LIMIT_OF_STEPS #Executing the heuristic until the limit or when it's done.
    N_SAMPLES = args['N_SAMPLES']

    min_obj = args['min_obj']
    objective = 1
    if not min_obj:
        objective = -1
    min_cost = math.inf*objective
    best_action = None

    # making a checkpoint to load to the start of the sampled trayectory
    initial_checkpoint = env.make_checkpoint()
    # Starting from a trayectory from the list
    for t in range(l_trayectories):
        trayectory = trayectories[t]
        total_cost = 0.0
        for _ in range(N_SAMPLES):
            ALPHA = alpha # Start the path with Alpha
            sample_cost = 0.0
            action_cost = 0.0
            # Executing the trayectory branch
            action = 0
            for action in range(l_trayectory): # Lookahead
                observation, action_cost, done, _ = env.step(trayectory[action])
                sample_cost += ALPHA * action_cost
                ALPHA = ALPHA * alpha
            # Executing the heuristic for k steps
            for _ in range(K):
                if done: # Here to chance to cancel the loop if necesary
                    break #The environment is terminated. Branch is closed.
                to_H = To_H.copy()
                to_H['observation'] = observation
                action_from_H = H(to_H)
                observation, action_cost, done, _ = env.step(action_from_H)
                # Extract heuristic cost
                sample_cost += ALPHA * action_cost
                ALPHA = ALPHA * alpha
            #End of the heuristic running
            # Termination value is estimated to 0 at the moment. 
            # Here could be an approximator from the last observationfrom the environment.
            sample_cost += 0
            # Adding the sample cost to the average cost of the samples
            total_cost += sample_cost / N_SAMPLES
            # Restarting the environment from the initial state.
            env.load_checkpoint(initial_checkpoint)
            # keeping the best only!
            if objective*total_cost < objective*min_cost:
                best_action = trayectory[0] # Saving the action with better sampled cost
                min_cost = total_cost
    # All trayectories sampled!
    del env # Closing the copy of the environment
    # It returns the best cost and action from its samples.
    return (best_action, min_cost)