# -*- coding: utf-8 -*-
"""
Created on jun 2020

@author: bobobert, MauMontenegro
"""

# Rob was here, so did Mau

# Math
import numpy as np
import math

# Parallel exec
from multiprocessing import Pool
import os

#Misc for looks and graphs.
import tqdm
import time
import matplotlib.pyplot as plt
import imageio
import pickle
import re

#Importing the sampler
from rollout_sampler import sample_trayectory, R_ITER

class Policy(object):
    """
    Object to create, save, manage and call a new policy. This type of policy uses a dict
    type to store the controls and its values for the states as the keys. A string system
    to represent states is recomended for ease to use with the hash table. This is usually
    referred as a tabular policy.

    Parameters
    ----------
    behavior : 'stochastic'(default), 'deterministic', 'max', 'min'. This is a variable to save the 
        prefered behavior of the policy when more than one control is available for a state
        when called.
    min_objective : Boolean, to specify when targeting the lower cost; if true, or the higer cost;
        if False, action control for a given state. If specified in behavior as 'min' or 'max' this
        is set accodingly. Default is True.
    DEBUG : Boolean. If you want to be printing the messages of the management of the policy
        or not.

    Methods
    -------
    Policy.new()
        Method to store for state key the control and value given. 

        Returns
        None

    Parameters
    ----------
    key : The state codification.
    control : control type
        Control chose when the state key is present. If the state has been already
        seen the control is appended to a list of controls. If the control itself has been
        already seen, its likehood to be chosen is increased proportionally.
    value : float
        Value associated with the control in that state. if the control has been already
        seen the value is averaged.

    Policy.call()
        Method to call a control from a given state key. Its behavior is by default the one given 
        when the object was created.

        Returns
        control type or False if key does not exists on the policy.

    Parameters
    ----------
    key : The state codification
    ovr : str
        (Optional, default given by the init behavior var) if one wishes to override the 
        default behavior, just pass an argument in this variable.
    """

    # Missing a method to save and read policy
    #Creates a new Policy    
    def __init__(self, behavior = 'stochastic', min_objective = True, DEBUG = False):
        self.policy = dict()
        self.debug = DEBUG
        self.behavior = behavior
        self.min_objective = min_objective
        self.keys = 0
        if not behavior == 'stochastic':
            self.min_objective = self.ridbuk_or_naik(behavior)
        # EVERYONE GETS A RANDOM GENERATOR! MUWAHAHAHAH
        self.rg = np.random.Generator(np.random.SFC64())
            
    @staticmethod
    def ridbuk_or_naik(behavior):
        # Function to determine what the string says
        if behavior == 'min' or behavior == 'Min' or behavior == 'MIN' or behavior == 'deterministic':
                return True
        elif behavior == 'max' or behavior == 'Max' or behavior == 'MAX':
            return False
        else:
            raise "Error with the behavior. {} was given, but is not valid.".format(behavior)

    #For every action-state calls this method to update policy    
    def new(self, key, control, value):
        #print(key)
        try:
            # The key state is already seen
            controls, values, actions_freq, total_freq, m_i = self.policy[key]
            # Iteration over the available controls in keys until reach actual control
            if control in controls:
                if self.debug:
                    print("Estado con acción ya escogida")
                    print(self.policy[key])
                i = 0
                for j in controls:
                    if j == control:
                        break
                    else:
                        i += 1
                #Update qvalue by averaging existing value for this state-action
                actions_freq[i] += 1   
                values[i] = ((actions_freq[i] - 1)*values[i] + value) / actions_freq[i]
            #If state has not this control, then add it with actual reward and update freq
                if self.min_objective and values[i] < values[m_i]:
                    m_i = i
                elif not self.min_objective and values[i] > values[m_i]:
                    m_i = i
            else:
                if self.debug:
                    print("Estado con acción aún no vista")
                    print(self.policy[key])
                controls.append(control)
                values.append(value)
                actions_freq.append(1)
                if self.min_objective and value < values[m_i]:
                    m_i = len(controls) - 1
                elif not self.min_objective and value > values[m_i]:
                    m_i = len(controls) - 1
            total_freq += 1
            self.policy[key] = (controls, values, actions_freq, total_freq, m_i)
            if self.debug: print(self.policy[key])
        except:
            # The key state is new, so their values are created
            #Set of controls, Set of q_values,freq of choosing this control, total freq of being in this state
            self.policy[key] = ([control],[value],[1],1,0)
            self.keys += 1
            if self.debug:
                print("Nuevo estado agregado")    
                print(self.policy[key])
    
    #Actual state of Policy        
    def __repr__(self):
        s = "Policy with {0} states \n |".format(self.keys)
        for i,key in enumerate(self.policy.keys()):
            s += "\n |-- State {0} controls:{1}".format(i, self.policy[key][0])
            s += "\n |     '-- Control objective {}".format(self.policy[key][0][self.policy[key][4]])
        s +="\n |\nEnd Policy\n"
        return s

    def __del__(self):
        print("Policy terminated!")
        return None
    
    #Return the best action to take given a key state. Two modes. 
    # include as override and store in __init__ a desired behavior.
    def call(self, key, ovr=None):
        if not ovr is None:
            mode = ovr
        else:
            mode = self.behavior
            
        try:             
            controls, _, freqs, total_freq, m_i = self.policy[key]
            #Select action based on a probability determined by its frequency and total frequency
            if mode == "stochastic":
                if self.debug: print("Seleccion estocastica de controles")
                freq_r = np.array(freqs) / total_freq
                return self.rg.choice(controls, 1, p=freq_r)[0]
            #Select action based only on best qvalue
            else: # mode=="deterministic": . RWH always return deterministic otherwise
                if self.debug: print("selección determinista de controles")
                return controls[m_i]
            #else:
                #return False
        except:
            # Key not in policy!
            return False

def Rollout(
    env, 
    H, 
    alpha=1, 
    epsilon=0, 
    K=-1,
    lookahead = 1, 
    N_samples=10, 
    n_workers=-1,
    H_args=None,
    min_objective=True,
    rg = None
    ):
    """Funtion to do a rollout from the actions available and the use of the Heuristic H.
    The argument k is to truncate how many steps the heuristic is going to execute.
    
    This function is wrote in base in the rollout algorithm as presented in the book of
    Dimitry Bertsekas Reinforcement Learning and Optimal Control.
    
    - BERTSEKAS, Dimitri P. Reinforcement learning and optimal control. Athena Scientific, 2019.

    It returns the best action from the rollout, and the expected value from taking it.
    action type, float.
    Parameters
    ----------
    env : Variable to reference the environmet to work with.
        It is expect to be an object class Helicopter for this version.
        The next methods and variables are expected in env:
        - action_set
        - step
        - copy()
        - make_checkpoint()
        - load_checkpoint()
    H : Object or function that references the heurist to execute. These type of functions
        requiere to support their inputs as a dict with at least the 'observation', and 'env'
        keys on it.
    H_args : Optional Dict of arguments to pass to the heuristic.
    alpha : Discount factor for the cost function.
    epsilon : It must be a quantity of probability in the range 0<epsilon<=1 to take an exploration
        action. This makes up the behaviour in a epsilon-greedy technique. In this case greedy 
        been the heuristic H.
    K : Integer of number of steps to keep executing the heuristic.
    lookahead : int
        Numbert of steps that the rollout algorithm can take in greedy form, forming a series of controls,
        that minimizes of maximizes the cost function. This increases the cost of the computation, and 
        discards all the controls but except the first.
    N_samples : Number of samples required to calculate the expected value of the
        cost function.
    n_workers : Quantity of threads that the function can use to sample the environment.
        If set to 1, a sequential execution will be done.
    min_objective : Bool variable to define if the objective is to maximize of minimize the cost function.
        This is problem-model dependant.
    rg : numpy.random.Generator type
        This function now needs an external random generator from the numpy library.
    """
    # Executive variables
    prune_samples = int(math.ceil(0.2*N_samples)) #20% of total samples goes to obtain controls and discard the worst
    NTOP=4 #Top 4 controls that are used in rest of samples
    cpus = n_workers
    parallel = False
    if cpus == 1:
        parallel = False
    elif cpus > 1:
        parallel = True
        if lookahead == 1:
            cpus = len(env.action_set) 
    else:
        raise "Something went wrong! {} is an invalid workers number!".format(cpus)
    
    objective = 1
    if not min_objective:
        # The default program is wrote to minimize.
        objective = -1
    # epsilon-greedy in action. 
    # Putting this in here comes handy to calculate the expected cost of the epsilon action
    # instead of sampling all the other possible trayectories. While using parallel is assigned.
    explotation = True
    if epsilon > rg.random(dtype=np.float32):
        # Epsilon-greedy behaviour
        # Turning to exploration
        explotation = False
    if explotation: #Explotation step. Sampling all the expected values to chose greedy accordingly to rollout algorithm. 
        # Creation of an iterable with the action set to make the samples
        # R_ITER could explode in resources comsumption if the lookahead is greater than 1
        # As it does a tree with Acion_set^lookahead branches to sample. 


        #Make an Iterable with some samples(prune_samples) to discard controls that are not promising.
        to_evaluate = R_ITER(env, H, cpus, alpha, K, lookahead, prune_samples, H_args=H_args,
                        Min_obj=min_objective)
               
        if parallel:
            # Execute the sampling in multiple threads of the cpu with map function
            # Usually this perfoms better than single thread for long trayectories/number of samples
            # If the trayectories are short, of the number of samples is little, a sequential run could perform better
            costs = [] 
            chunk_trayectory= []
            new_trayectories= []
            #Sample few times all trajectories
            p = Pool(processes=cpus)                      
            costs = p.imap_unordered(sample_trayectory,to_evaluate)                     
            p.close()
            p.join()            
            #Get only the best NTOP controls after some iterations based on the average cost
            best_controls = sorted(costs, key = lambda x: x[1], reverse = False)[:NTOP]           
            best_controls = [x[0] for x in best_controls]          
            #Create new chunk size according to new controls
            new_chunk= math.ceil(9*len(best_controls) / cpus)          
            #Copy only trayectories that belong to selected controls and create
            #the pooling structure according to cpus.            
            i=0 
            for element in to_evaluate.all_trayectories:
                for branch in element:
                    for ctrl in best_controls:                    
                        if ctrl == branch[0]:
                            if i < new_chunk:
                                chunk_trayectory+= [branch]   
                                i+=1   
                            if i==new_chunk:
                                new_trayectories+=[chunk_trayectory]  
                                chunk_trayectory=[]
                                i=0
            new_trayectories+=[chunk_trayectory]  
            if lookahead==1:
                new_trayectories=to_evaluate.all_trayectories             
            #Setting new values to iterable(Creating a new one will consume much time)
            to_evaluate.all_trayectories=new_trayectories 
            to_evaluate.j=0
            to_evaluate.N_samples=N_samples- prune_samples    
            costs = [] 
            p = Pool(processes=cpus)                      
            costs = p.imap_unordered(sample_trayectory,to_evaluate)
        else:
            # Execute an evaluation at a time in one cpu thread
            chunk_trayectory= []
            new_trayectories= []
            costs = []
            for i in to_evaluate:
                costs.append(sample_trayectory(i))          
            #Get only the best NTOP controls after some iterations based on the average cost
            best_controls = sorted(costs, key = lambda x: x[1], reverse = False)[:NTOP] 
            best_controls = [x[0] for x in best_controls]  
            #Create new chunk size according to new controls
            new_chunk= math.ceil(9*len(best_controls) / cpus)          
            #Copy only trayectories that belong to selected controls
            i=0 
            for element in to_evaluate.all_trayectories:
                for branch in element:
                    if branch[0] in best_controls:
                        if i < new_chunk:
                            chunk_trayectory+= [branch]   
                            i+=1   
                        if i==new_chunk:
                            new_trayectories+=[chunk_trayectory]  
                            chunk_trayectory=[]
                            i=0
            new_trayectories+=[chunk_trayectory]           
            #Setting new values to iterable(Creating a new one will consume much time)
            to_evaluate.all_trayectories=new_trayectories 
            to_evaluate.j=0
            to_evaluate.N_samples=N_samples- prune_samples  
            #Get Costs for bests controls
            costs = []
            for i in to_evaluate:
                costs.append(sample_trayectory(i))            
        # costs is the list of the costs of the sampled trayectories from their respective action
        # As the greedy nature of the algorithm, the sample trayectory do only returns the first control
        # and the expected cost. Here we take that minimum not caring of the rest of the trayectory may have
        # been.               
        min_cost, r_action = objective*np.inf, None             
        for action, cost in costs:                                
            if objective*cost < objective*min_cost:
                r_action = action # Saving the action with less cost
                min_cost = cost         
        if parallel: # Terminating the pool of processes in this rollout
            p.close()
            p.join()
        del to_evaluate # Cleaning the iterable 
        return r_action, min_cost
        
    else: # Option to take an exploration step
        actions = list(env.action_set)
        e_action = rg.choice(actions) # Chose a random action.
        e_actions = [e_action for _ in range(N_samples)]
        # Trying to use an efective way to sample the trayectory cost
        # It creates a worker to do a simple sample of the trayectory, instead 
        # of doing all the samples in a simple worker.
        to_evaluate = R_ITER(env, H, cpus, alpha, K, 1, 1, H_args=H_args, Action_set=e_actions,
                        Min_obj=min_objective)
        if parallel:
            p = Pool(processes=cpus)
            costs = p.imap_unordered(sample_trayectory, to_evaluate)
            p.close()
            p.join()
        else:
            costs = []
            for i in to_evaluate:
                costs.append(sample_trayectory(i))
        e_cost = 0
        for _, cost in costs:
            e_cost += cost / N_samples
        del to_evaluate
        return e_action, e_cost

class Experiment():
    """
    Class design to run a complete rollout experiment with the options to generate graphs,
    animations, save results in pickle form. 
    As this is a RL-Rollout implementation it needs a base heuristic or base policy to 
    call and to compare to start improving the min/max of the cost function.
    
    Parameters
    ----------
    ENV : Environment Object
        Reference to the environment object. This funcion is constructed to the EnvMakerForestFire
        environment.
        The next methods and variables are expected in env:
        - action_set
        - step()
        - Encode()
        - copy()
        - make_checkpoint()
        - load_checkpoint()
        - frame()
        - make_gif()

    H : Function object
        Object or function that references the heurist to execute. These type of functions
        requiere to support their inputs as a dict with at least the 'observation', and 'env'
        keys on it. It must return an action type.

    H_ARGS : *Dict type 
        Optional Dict of additional arguments to pass to the heuristic.

    PI : Policy object
        Here one can pass an policy already started. If None it generates a new one.

    N_TRAIN : int
        Number of tests to run the experiment  with .run() for constructing 
        an unique policy with rollout.
        The result of this is the average of the costs between all the tests for each run.
        Notice that inside a run every test starts with the same initial state.
        Each test has execute all the follwing variables. Then for a sequential
        run the complexity of the execution roughly speaking is above of (N_TEST * N_STEPS * N_SAMPLES * K).
        To lower the time of execution increase the number or workers as posible as it parallelize the
        N_SAMPLES // N_WORKERS. 

    N_STEPS : int 
        Number of steps that the environment takes. Speaking about the Helicopter environment, the variable
        freeze has an effect to update the environment each FREEZE steps. Therefore, the agent in total execute
        N_STEPS * FREEZE steps.

    N_SAMPLES : int
        Number of samples required to calculate the expected value of the
        cost function.

    K : int
        Number of steps to keep executing the heuristic.

    LOOKAHEAD : int
        Numbert of steps that the rollout algorithm can take in greedy form, forming a series of controls,
        that minimizes of maximizes the cost function. This increases the cost of the computation, and 
        discards all the controls but except the first.

    ALPHA : float 
        Discount factor for the cost function.

    EPSILON : float
        It must be a quantity of probability in the range 0<epsilon<=1 to take an exploration
        action. This makes up the behaviour in a epsilon-greedy technique. In this case greedy 
        been the heuristic H.

    EPSILON_DECAY : float
        The rate between 0 and 1, in which the value of epsilon decays every time it is used on the
        rollout executions.

    MIN_OBJECTIVE : bool 
        Variable to define if the objective is to maximize of minimize the cost function.
        This is problem-model dependant.

    N_WORKERS : int 
        Quantity of threads that the function can use to sample the environment.
        If set to 1, a sequential execution will be done on a single thread.

    RUN_GIF : bool
        Variable to control the behavior if the last execution of run generates frame for each
        agent step for being able to generate a .gif with the .gif method.

    Methods
    -------
    Experiment.run()

    Experiment.policy_test()

    Experiment.make_graph()
        
    Experiment.make_gif()
        
    Experiment.pickle()
        Dump the sequence of costs obtainen and the policy object

    Experiment.reset()
        Cleans buffers for graphs and gifs. Restart countes. 

    """
    def __init__(
        self,
        ENV,
        H,
        H_ARGS = None,
        PI = None,
        N_TRAIN = 10,
        N_STEPS = 25,
        N_SAMPLES = 29,
        K = 100,
        LOOKAHEAD = 1,
        ALPHA = 0.99,
        EPSILON = 0,
        EPSILON_DECAY = 0.99,
        MIN_OBJECTIVE = True,
        N_WORKERS = -1,
        RUN_GIF = False,
        ):
        def check_ints(suspect):
            assert (suspect >= 1) and isinstance(suspect, int),\
                "This number must an integer of at least 1. {} = {} given instead.".format(type(suspect), suspect)
            return suspect
        def check_prob(suspect):
            assert (suspect <= 1) and (suspect >= 0),\
                "This value must be between 0 and 1. {} was given".format(ALPHA)
            return suspect
        # Saving references to objects and classes.
        self.env = ENV
        self.env_h = None # Var to env copy for applying the heuristic
        self.H = H
        if H_ARGS is None:
            self.H_args = dict()
        else:
            self.H_args = H_ARGS
        self.min_obj = MIN_OBJECTIVE
        assert isinstance(MIN_OBJECTIVE, bool), "With a True/False indicate if minimize is the objective. Invalid type {} passed".format(type(MIN_OBJECTIVE))
        if PI is None:
            # Creates a new policy
            self.PI = Policy(min_objective=MIN_OBJECTIVE)
        else:
            self.PI = PI
        # Loading variables
        self.N_TRAIN = check_ints(N_TRAIN)
        self.N_STEPS = check_ints(N_STEPS)
        self.N_SAMPLES = check_ints(N_SAMPLES)
        if K < 0:
            self.K = -1
        else:
            self.K = check_ints(K)
        self.LOOKAHEAD = check_ints(LOOKAHEAD)
        self.alpha = check_prob(ALPHA)
        self.epsilon_op = check_prob(EPSILON)
        self.epsilon = check_prob(EPSILON)
        self.epsilon_decay = check_prob(EPSILON_DECAY)

        self.init_logger = False
        self.init_logger = self.logger("Logger initialized.",False) 

        # This class has its own random generator.
        self.rg = np.random.Generator(np.random.SFC64())
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []
        self.c_runs = 0
        self.theres_run_gif = False
        self.theres_test_gif = False
        self.RUN_GIF = RUN_GIF
        self.frames_run_r = []
        self.frames_run_h = []
        self.frames_test_r = []

        self.mod = "Cost"

        cpu_sys = os.cpu_count()
        if N_WORKERS == 1:
            self.cpus = 1
        elif N_WORKERS == -1:
            self.cpus = cpu_sys # ALL YOUR BASE ARE BELONG TO US, with SMT by default
        elif N_WORKERS < 0 or N_WORKERS > cpu_sys:
            self.logger("N_WORKERS = {} is not valid. The computer has up to {} threads only.".format(N_WORKERS, cpu_sys))
            self.cpus = cpu_sys // 2  # half of thread given, bye SMT
        else:
            self.cpus = N_WORKERS
        self.logger("Assigning {} threads of the CPU for experiment workers.".format(self.cpus))
    
    def __del__(self):
        self.logger("The experiment is OVER!")
        self.logfile.close()
        del self.env_h
        del self.env
        del self.PI
        return None

    def run(self, GIF=None, GRAPH=True):
        """
        Creates an initial state from reseting the environment and runs all the number of train
        iterations and so on. This updates the policy with more states or with better actions.

        Parameters
        ----------
        GIF : bool 
            Variable to indicate if you desired to generate frames for the last 
            train loop of the run, if the class was initialized with this behavior on this one
            changes nothing. Default False.
        GRAPH : bool
            Draws and saves the graphs from the experiment. If there's not a graph generated and
            one does not restarts the class 
        """
        if not GIF is None:
            RUN_GIF = GIF
        else:
            RUN_GIF = self.RUN_GIF
        self.logger("Run {} - Metadata: alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{}".format(
            self.c_runs, self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES), time=False)
        self.logger(" |")
        # Reseting env and storing the initial observations
        observation = self.env.reset()
        observation_h = observation
        #Making copy of the env to apply the heuristic
        self.env_h = self.env.copy()
        # Making checkpoints
        checkpoint_env = self.env.make_checkpoint()
        checkpoint_env_h = self.env_h.make_checkpoint()
        # Lists to save the results from the N_TRAIN
        RO_RESULTS=[]
        H_RESULTS=[]
        RO_RESULTS_C=[]
        H_RESULTS_C=[]
        # Measuring time of execution. 
        start = time.time()
        # First loop to execute an rollout experiment.
        for n_test in range(self.N_TRAIN):
            # In order to compare the advance between the two environments
            # their random generator is reseeded with the same seed.
            # This ones should advance equally on the all the run, but the samples
            # as they are copies they generate a new random gen, so the samples wont suffer
            # from this
            M_SEED = int(self.rg.random()*10**4)
            self.env.rg = np.random.Generator(np.random.SFC64(M_SEED))
            self.env_h.rg = np.random.Generator(np.random.SFC64(M_SEED))    
            self.logger(" |-- Test : {} of {}".format(n_test+1, self.N_TRAIN))
            # Making a checkpoint from the initial state generated.         
            self.env.load_checkpoint(checkpoint_env)
            self.env_h.load_checkpoint(checkpoint_env_h)
            # Setting up vars to store costs
            rollout_cost=0
            heuristic_cost=0
            rollout_cost_step=[]
            heuristic_cost_step=[]
            # Making the progress bar
            bar = tqdm.tqdm(range(self.env.moves_before_updating * self.N_STEPS), miniters=0)
            for i in bar:
                #Calls Rollout Strategy and returns action,qvalue
                env_state = self.env.Encode() # The observed state encoded
                r_action, q_value = Rollout(
                    self.env, 
                    self.H,
                    alpha=self.alpha,
                    epsilon=self.epsilon,
                    K=self.K,
                    lookahead=self.LOOKAHEAD,
                    N_samples=self.N_SAMPLES,
                    H_args = self.H_args,
                    n_workers = self.cpus,
                    min_objective=self.min_obj,
                    rg=self.rg)
                #Update epsilon it goes from stochastic to deterministic 
                self.epsilon = self.epsilon * self.epsilon_decay
                #Calls Heuristic and return best action
                To_H = self.H_args.copy()
                To_H['env'] = self.env_h
                To_H['observation'] = observation_h
                h_action = self.H(To_H)
                #Update Policy        
                self.PI.new(env_state, r_action, q_value)
                #Helicopter take an action based on Rollout strategy and heuristic
                observation, ro_cost, _, _ = self.env.step(r_action)
                observation_h, h_cost, _, _ = self.env_h.step(h_action)
                if RUN_GIF and (n_test == self.N_TRAIN - 1):
                    # Framing just the last round
                    self.env.frame(title="Rollout step {}-th".format(i))
                    self.env_h.frame(title="Heuristic step {}-th".format(i))
                #Update Rollout Total cost
                rollout_cost += ro_cost  #Acumulative cost for rollout          
                rollout_cost_step.append(rollout_cost)  #List of cost over time
                #Update Heuristic Total cost
                heuristic_cost += h_cost
                heuristic_cost_step.append(heuristic_cost)
                #self.logger("Sync" if self.env.rg.random() == self.env_h.rg.random() else 'NotSync')
                #Generate a message
                msg =    " |   |      |"
                msg += "\n |   |      |-- Agent step {}".format(i)
                msg += "\n |   |      |   |-- Rollout with action {} and cost : {}".format(r_action, ro_cost)
                msg += "\n |   |      |   '-- Heuristic with action {} and cost : {}".format(h_action, h_cost)
                bar.write(msg)
                self.logger(msg, False, False)
            bar.close()
            msg =    " |   |"
            msg += "\n |   |-- Test {} results".format(n_test+1)
            msg += "\n |       |-- Total Rollout cost : {}".format(rollout_cost)
            msg += "\n |       '-- Total Heuristic cost : {}".format(heuristic_cost)
            msg += "\n |"
            self.logger(msg)
            #Costs p/test
            RO_RESULTS.append(rollout_cost)
            H_RESULTS.append(heuristic_cost)
            #Cumulative costs p/test
            RO_RESULTS_C.append(rollout_cost_step)
            H_RESULTS_C.append(heuristic_cost_step)
        msg = " | Run {} done.".format(self.c_runs)
        msg+= "\nMetadata: alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{}".format(
                self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES)
        self.logger(msg, time=False)
        self.logger("Total run time %.2f s"%(time.time()-start))

        # Saving to the class
        self.runs_rollout_results += RO_RESULTS
        self.runs_rollout_results_step += RO_RESULTS_C
        self.runs_heu_results += H_RESULTS
        self.runs_heu_results_step += H_RESULTS_C
        if GRAPH:
            self.make_graph(title_head='Run {}'.format(self.c_runs), mod=self.mod)
        self.c_runs += 1
        # Saving data to generate gif
        if RUN_GIF: 
            self.frames_run_r += self.env.frames
            self.env.frames = []
            self.frames_run_h += self.env_h.frames
            self.env_h.frames = []
            self.theres_run_gif = True
        
        return None

    def reset(self):
        # Free memory.
        self.env.checkpoints = []
        self.env_h.checkpoints = []
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []
        self.frames_run_r = []
        self.frames_run_h = []
        self.frames_test_r = []
        self.theres_run_gif = False
        self.theres_test_gif = False
        self.c_runs = 0

    
    def policy_test(self, N_TEST=5, N_STEPS=20, PI_MODE=None, EPSILON=None, GIF=True,
    GRAPH=True):
        """
        Method to run the policy in its state so far in the environment of the experiment.
        If the policy does not have the state on it, it applies epsilon-greedy with the
        greedy behavior being the use of the heuristic loaded.

        Parameters
        ----------
        N_TEST : int
            Number of tests to run the generated policy object inside the environment env.
        N_STEPS : int
            Number of steps that the environment takes along the test.
        PI_MODE : str or None
            If set to None the policy uses its default behavior "stochastic". Otherwise, 
            it passes the mode as override. 
        EPSILON: float or None
            Epsilon value between 0 and 1. If None (Default), it usese the EPSILON given 
            at the begining.
        GIF: bool
            If true, it generates a file .gif for the best. This by default set to true.
        GRAPH : bool
            If set to true draws and save the graphs for the experiment results. This uses the
            same buffer as the run graphs. If you skipped graphs on run the results will be erased.
        """
        # based on RUN
        if not GIF is None:
            RUN_GIF = GIF
        else:
            RUN_GIF = self.RUN_GIF
        self.logger("Testing policy. Metadata: epsilon-{} k-{} LH-{} n_test-{} n_steps-{}".format(
                EPSILON, self.K, self.LOOKAHEAD, N_TEST, N_STEPS), time=False)
        self.logger(" |")
        # Reseting env and storing the initial observations
        observation = self.env.reset()
        observation_h = observation
        #Making copy of the env to apply the heuristic
        self.env_h = self.env.copy()
        # Making checkpoints
        #checkpoint_env = self.env.make_checkpoint()
        #checkpoint_env_h = self.env_h.make_checkpoint()
        # Lists to save the results from the N_TRAIN
        RO_RESULTS=[]
        H_RESULTS=[]
        RO_RESULTS_C=[]
        H_RESULTS_C=[]
        pi_calls = 0
        calls = 0
        # Measuring time of execution. 
        start = time.time()
        # First loop to execute an rollout experiment.
        for n_test in range(N_TEST):
            M_SEED = int(self.rg.random()*10**4)
            self.env.rg = np.random.Generator(np.random.SFC64(M_SEED))
            self.env_h.rg = np.random.Generator(np.random.SFC64(M_SEED))
            if EPSILON is None:
                epsilon = self.epsilon_op
            else:
                epsilon = EPSILON        
            self.logger(" |-- Test : {} of {}".format(n_test+1, N_TEST))
            # Making a checkpoint from the initial state generated.         
            #self.env.load_checkpoint(checkpoint_env)
            #self.env_h.load_checkpoint(checkpoint_env_h)
            # Setting up vars to store costs
            rollout_cost=0
            heuristic_cost=0
            rollout_cost_step=[]
            heuristic_cost_step=[]
            # Making the progress bar
            bar = tqdm.tqdm(range(self.env.moves_before_updating * self.N_STEPS), miniters=0)
            for i in bar:
                # Calling the policy with the actual environment state
                env_state = self.env.Encode() # The observed state encoded
                action = self.PI.call(env_state, ovr=PI_MODE)
                if action is False:
                    # The state is not in the policy. An Epsilon-greedy is executed.
                    if epsilon >= self.rg.random(dtype=np.float32):
                        # Exploration is ON
                        action = self.rg.choice(self.env.action_set)
                        s = "Exploration Step"
                    else:
                        # Explotation with H
                        To_H = self.H_args.copy()
                        To_H['env'] = self.env
                        To_H['observation'] = observation
                        action = self.H(To_H)
                        s = "Greedy Step"
                else:
                    s = "Policy Step"
                    pi_calls += 1
                calls += 1
                #Update epsilon it goes from stochastic to deterministic 
                epsilon = epsilon * self.epsilon_decay
                #Calls Heuristic and return best action
                To_H = self.H_args.copy()
                To_H['env'] = self.env_h
                To_H['observation'] = observation_h
                h_action = self.H(To_H)
                #Helicopter take an action based on Rollout strategy and heuristic
                observation, ro_cost, _, _ = self.env.step(action)
                observation_h, h_cost, _, _ = self.env_h.step(h_action)
                if RUN_GIF:
                    title = "Test {} - ".format(n_test + 1)
                    title += s
                    self.env.frame(title=title)
                #Update Rollout Total cost
                rollout_cost += ro_cost  #Acumulative cost for rollout          
                rollout_cost_step.append(rollout_cost)  #List of cost over time
                #Update Heuristic Total cost
                heuristic_cost += h_cost
                heuristic_cost_step.append(heuristic_cost)
                #Generate a message
                msg =    " |   |      |"
                msg += "\n |   |      |-- Agent step {}".format(i)
                msg += "\n |   |      |   |-- Rollout with action {} and cost : {}. Mode {}".format(action, ro_cost, s)
                msg += "\n |   |      |   '-- Heuristic with action {} and cost : {}".format(h_action, h_cost)
                bar.write(msg)
                self.logger(msg, False, False)
            bar.close()
            msg =    " |   |"
            msg += "\n |   '-- Test {} results".format(n_test+1)
            msg += "\n |       |-- Total Rollout cost : {}".format(rollout_cost)
            msg += "\n |       '-- Total Heuristic cost : {}".format(heuristic_cost)
            msg += "\n |"
            self.logger(msg)
            #Costs p/test
            RO_RESULTS.append(rollout_cost)
            H_RESULTS.append(heuristic_cost)
            #Cumulative costs p/test
            RO_RESULTS_C.append(rollout_cost_step)
            H_RESULTS_C.append(heuristic_cost_step)
            # Reseting the environment
            observation = self.env.reset()
            obvservation_h = self.env_h.reset()
        msg = " | Test done. Metadata: epsilon-{} k-{} LH-{} n_test-{} n_steps-{}".format(
                EPSILON, self.K, self.LOOKAHEAD, N_TEST, N_STEPS)
        self.logger(msg, time=False)
        self.logger("Percentage of successful calls to policy: %.2f"%(pi_calls/calls*100), time=False)
        self.logger("Total run time %.2f s"%(time.time()-start))
        # Saving to the class
        self.runs_rollout_results += RO_RESULTS
        self.runs_rollout_results_step += RO_RESULTS_C
        self.runs_heu_results += H_RESULTS
        self.runs_heu_results_step += H_RESULTS_C
        if GRAPH:
            t_head = 'Test - %.2f pi success'%(pi_calls/calls)
            self.make_graph(title_head=t_head, mod=self.mod)
        # Saving to generate the GIF
        if RUN_GIF: 
            self.frames_test_r += self.env.frames
            self.env.frames = []
            self.theres_test_gif = True


    def make_graph(self, title_head='',mod='Cost',dpi=120):
        """
        Method to graph and save two graphs per expriment. An average cost 
        per step on all the test. And a progession of average cost of the per 
        experiment.
        Each time that is called, it eareses the buffers

        Parameters
        ----------
        tittle_head : str
            Puts a Head on the tittle of the images. It prints on both the graph 
            and the file.
        mod : str
            Modifier for calling the cost or reward.
        """

        Experiment.check_dir("Runs")
        time_s = Experiment.time_str()

        # First graph - Avg. Cost per step of all test
        # Acumulative cost per step per test
        R_RESULTS_STEPS = np.array(self.runs_rollout_results_step) 
        H_RESULTS_STEPS = np.array(self.runs_heu_results_step)
        x = range(R_RESULTS_STEPS.shape[1])
        plt.plot(x, np.mean(R_RESULTS_STEPS, axis=0), label='Rollout')
        plt.plot(x, np.mean(H_RESULTS_STEPS, axis=0), label='Heuristic')
        plt.xlabel('Step')
        plt.ylabel('Average '+mod)
        if title_head != '':
            plt.title('{} - Rollout results average {}/Step'.format(title_head,mod))
        else:
            plt.title('Rollout results average tests {}/Step'.format(mod))
        plt.legend()
        plt.savefig(
            "./Runs/Rollout avg cost-step alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{} -- {}.png".format(
                self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES,
                time_s))
        plt.clf() # cleaning figure
        # Doing graph cost per test.
        # Cost per test
        R_RESULTS_TEST = np.array(self.runs_rollout_results)
        H_RESULTS_TEST = np.array(self.runs_heu_results) 
        x = range(1, len(self.runs_rollout_results)+1)
        plt.plot(x, R_RESULTS_TEST, label='Rollout')
        plt.plot(x, H_RESULTS_TEST, label='Heuristic')
        plt.xlabel('Test')
        plt.ylabel(mod)
        if title_head != '':
            plt.title('{} - Rollout results {}/Test'.format(title_head,mod))
        else:
            plt.title('Rollout results {}/step'.format(mod))
        plt.legend()
        plt.savefig(
            "./Runs/Rollout cost-test alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{} -- {}.png".format(
                self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES,
                time_s))
        plt.clf()
        # Clean the buffers
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []
        return None

    def make_gif(self, RUN=False, TEST=False, fps=5):
        """
        Make and save .gif files from all the agent steps done in the environment with 
        the rollout and heuristic choices in the last train of the last run done.
        For this function to work is necessary to indicate on the initial vairbales 
        RUN_GIF = True, to to check it when about to make a .run(GIF=True)
        
        Parameters
        ----------
        RUN : bool
            If True, generates a .gif from the rollout agent from the last run. It's the last due to resouces
            management.
        TEST : bool
            If True, and a .policy_test() has been executed, then it generates the gif for the best run test.
        """
        Experiment.check_dir("Runs")
        time_s = Experiment.time_str()
        if RUN and (self.theres_run_gif):
            self.logger("Creating gif for runs. This may take a while.")
            imageio.mimsave("./Runs/Helicopter Rollout Run alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{} -- {}.gif".format(
                self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES, time_s), 
                self.frames_run_r, fps=fps)
            self.frames_run_r = []
            heu_par = ''
            for arg in self.H_args.keys():
                heu_par += ' ' + str(arg) + '-' + str(self.H_args[arg])
            imageio.mimsave("./Runs/Helicopter Heuristic Run --{} -- {}.gif".format(heu_par,time_s),
                self.frames_run_h, fps=fps)
            self.frames_run_h = []
            self.theres_run_gif = False
            self.logger("Run gif. Done!\n")
        if TEST and self.theres_test_gif:
            self.logger("Creating gif for tests. This may take a while.")
            imageio.mimsave("./Runs/Helicopter Rollout Test alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{} -- {}.gif".format(
                self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES, time_s),
                self.frames_test_r, fps=fps)
            self.frames_test_r = []
            self.theres_test_gif = False
            self.logger("Test gif. Done!\n")
        return None
    
    def save_policy(self, name_out=None):
        """
        Saves the actual policy object in a pickle dump file with extension .pi

        Parameters
        ----------
        name_out: str
            Name for the file, if default None then it generates a unique name
        """
        self.check_dir("Policies")
        name = name_out
        if name is None:
            # Default name scheme
            name = "Policy_Rollout Runs-{} alpha-{} epsilon-{} epsilon_decay-{} k-{} LH-{} n_samples-{} -- {}".format(
                self.c_runs ,self.alpha, self.epsilon_op, self.epsilon_decay, self.K, self.LOOKAHEAD, self.N_SAMPLES, self.time_str())
        file_h = open("./Policies/"+name+".pi",'wb')
        pickle.dump(self.PI,file_h)
        file_h.close()
        self.logger("Policy saved under name {}\n".format(name))
    
    def load_policy(self, name_in=None, dir="./Policies"):
        """
        Loads a policy object from a pickle dump file with extension .pck

        Parameters
        ----------
        name_in: str
            Name for the file, if default None then it looks in the actual direction
            for matching files and asks to chose one if available.
        dir: str
            If one wants to loook in another direction.
        """
        op_dir = os.getcwd()
        if dir != "":
            os.chdir(dir)
        if name_in is None:
            ls = os.listdir()
            options = []
            for s in ls:
                if re.match(".+\.pi$",s):
                    options += [s]
            if len(options) > 0:
                print("Please chose from the options")
                for i,s in enumerate(options):
                    print("| -- {} : {}\n\n".format(i,s))
            else:
                print("Folder {} has no .pi options".format(dir))
                os.chdir(op_dir)
                return None
            while True:
                choice = input("Enter option:")
                choice = int(choice)
                if choice < len(options):
                    break
            file_handler = open(options[choice], 'rb')
            self.PI = pickle.load(file_handler)
            file_handler.close()
            self.logger("Policy {} loaded.\n".format(options[choice]))
        else:
            file_handler = open(name_in, 'rb')
            self.PI = pickle.load(file_handler)
            file_handler.close()
            self.logger("Policy {} loaded.\n".format(name_in))
        print(self.PI)

    def logger(self, msg, prnt=True, time=True):
        """
        Just a method to save most of the print information in a file while the Experiment
        exists.

        Parameters
        ----------
        msg: str
            The string generated to log.
        prnt: bool
            If true a message will be displayed.
        time: bool
            If true a time stamp will be added to the message in the log.
        """
        if self.init_logger == False:
            self.check_dir("Logs")
            self.logfile = open("./Logs/rollout_log_{}.txt".format(self.time_str()),'wt')
            if prnt:
                print(msg,end='\n')
            if time:
                msg += " -- @ {}".format(self.time_str())
            self.logfile.write(msg+"\n")
            return True
        else:
            if prnt:
                print(msg,end="\n")
            if time:
                msg += " -- @ {}".format(self.time_str())
            self.logfile.write(msg+"\n")   

    @staticmethod
    def time_str():
        return time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())

    @staticmethod
    def check_dir(dir):
        assert isinstance(dir, str), "dir argument must be a string"
        ls = os.listdir(os.getcwd())
        if not dir in ls:
            print("Creating a folder {}".format(dir))
            os.mkdir(dir)
        time.sleep(1)