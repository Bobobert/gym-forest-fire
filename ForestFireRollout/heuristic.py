# -*- coding: utf-8 -*-
"""
Set of Heuristics 2.0

Created on jun 2020 v1.0

Modified on jun 2020 v2.0
    Added Calculate_Fire_Coefficient: An Heuristic that calculates
    for each zone a coefficient based on fire cells(+1), tree cells(+ p_fire)
    and empty cells (+ p_tree) and divided by grid total cells.
    Now args contain a 'mode', 1 for fire cells heuristic and 2  for coefficient heuristic.

@author: MauMontenegro
"""
import numpy as np
import random

def Heuristic(args):
    """
    This function is an heuristic with vision around the agent.
    The argumentes should be pass as a dictionary. 
    
    Example
    -------
    >>> action = Heuristic({
    >>>     'env':Forest_fire,
    >>>     'observation': Forest_fire.step(action),
    >>>     'vision':1,
    >>>     })

    args - dict
    -----------
    'env' : expecting helicopter environment object.
    'observation' : tuple (obs, reward, terminated, info)
        The observation emited by the environment
        where one wants to calculate the action with this heuristic
    'vision' : int
        Vision ahead the current cell of the agent on the environment.
    """
    assert isinstance(args, dict), "This function requieres a dict of kwards to perform. Please provide one."
    expected_args = ['env','observation','vision']
    for arg in expected_args:
        if args.get(arg) is None:
            raise "At least the argument {} is missing! Please do pass correct arguments".format(arg)
    env = args['env']
    vision = args['vision']
    mode = args['mode']
    grid, pos, remain_steps = args['observation']
    
    #Add borders in Grid perimeter according to vision range of helicopter, in order
    #to explore without boundary limits (Example: 2 vision add 2 cells at each side of Grid)
    Pad_grid=ExpandGrid(grid,vision)    
    
    #Get neighborhood in agent current position and vision range
    neighborhood= get_neighborhood(Pad_grid,pos,vision) 
    
    #Count fire cells or fire coefficient by zone(8 zones)    
    burned_densities={}
    
    #Up Zone
    up_zone=neighborhood[ 0:neighborhood.shape[0]-(vision+1),0:neighborhood.shape[1]] #Get Up Zone
    if mode==1: #Only count fire cells in a zone
        up_burned=Count_Burned_Trees(env,up_zone)  
    elif mode==2: #Calculate a coefficient of fire probability based on all cells
        up_burned=Calculate_Fire_Coefficient(env,up_zone) 
    burned_densities["up"]=up_burned #Add zone and fire density to dictionary
    
    #Up Left Zone
    up_left_zone=neighborhood[ 0:neighborhood.shape[0]-(vision+1),0:neighborhood.shape[0]-(vision+1) ]
    if mode==1:
        up_left_burned=Count_Burned_Trees(env,up_zone)
    elif mode==2:
        up_left_burned=Calculate_Fire_Coefficient(env,up_zone) 
    burned_densities["up_left"]=up_left_burned    
    
    #Up Right Zone
    up_right_zone=neighborhood[ 0:neighborhood.shape[0]-(vision+1),neighborhood.shape[0]-vision:neighborhood.shape[0] ]
    if mode==1:
        up_right_burned=Count_Burned_Trees(env,up_right_zone)
    elif mode==2:
        up_right_burned=Calculate_Fire_Coefficient(env,up_right_zone)       
    burned_densities["up_right"]=up_right_burned    
    
    #Down Zone
    down_zone=neighborhood[ neighborhood.shape[0]-vision:neighborhood.shape[0],0:neighborhood.shape[1]]
    if mode==1:
        down_burned=Count_Burned_Trees(env,down_zone) 
    elif mode==2:
        down_burned=Calculate_Fire_Coefficient(env,down_zone)      
    burned_densities["down"]=down_burned
    
    #Down Left
    down_left_zone=neighborhood[ neighborhood.shape[0]-vision:neighborhood.shape[0], 0:neighborhood.shape[0]-(vision+1) ]
    if mode==1:
        down_left_burned=Count_Burned_Trees(env,down_left_zone)
    elif mode==2:
        down_left_burned=Calculate_Fire_Coefficient(env,down_left_zone)       
    burned_densities["down_left"]=down_left_burned    
    
    #Down Right
    down_right_zone=neighborhood[ neighborhood.shape[0]-vision:neighborhood.shape[0], neighborhood.shape[0]-vision:neighborhood.shape[0] ]
    if mode==1:
        down_right_burned=Count_Burned_Trees(env,down_right_zone)
    elif mode==2:
        down_right_burned=Calculate_Fire_Coefficient(env,down_right_zone)      
    burned_densities["down_right"]=down_right_burned   
    
    #Left Zone
    left_zone=neighborhood[ 0:neighborhood.shape[0],0:neighborhood.shape[0]-(vision+1)]
    if mode==1:
        left_burned=Count_Burned_Trees(env,left_zone)  
    elif mode==2:
        left_burned=Calculate_Fire_Coefficient(env,left_zone)     
    burned_densities["left"]=left_burned
    
    #Right Zone
    right_zone=neighborhood[ 0:neighborhood.shape[1],neighborhood.shape[0]-vision:neighborhood.shape[0]]
    if mode==1:
        right_burned=Count_Burned_Trees(env,right_zone)
    elif mode==2:
        right_burned=Calculate_Fire_Coefficient(env,right_zone)      
    burned_densities["right"]=right_burned
    
    #Action based on burned trees/zone
    actions= ((1,2,3),
              (4,5,6),
              (7,8,9))
    
    #Max function will return a (key,value) tuple of the maximum value from the dictionary
    mx_tuple = max(burned_densities.items(),key = lambda x:x[1]) 
    #Mx_tuple[1] indicates maximum dictionary items value
    max_list =[i[0] for i in burned_densities.items() if i[1]==mx_tuple[1]] 
    
    #Apply Heuristic Rules according to fire cells in each zone
    #If there are more than 1 max burn zone, choose randomly
    if len(max_list) > 1: 
        a=random.choice(max_list)
        if a=="up":
            action=actions[0][1]
        elif a=="down":
            action=actions[2][1]
        elif a=="left":
            action=actions[1][0]
        elif a=="right":
            action=actions[1][2]
        elif a=="up_left":
            action=actions[0][0]
        elif a=="up_right":
            action=actions[0][2]
        elif a=="down_left":
            action=actions[2][0]
        elif a=="down_right":
            action=actions[2][2]
    #If there is only one zone with max fire density (move in up,down,right,left or corners only)
    elif len(max_list)==1:
        if max_list[0]=="up":
            action=actions[0][1]
        elif max_list[0]=="down":
            action=actions[2][1]
        elif max_list[0]=="left":
            action=actions[1][0]
        elif max_list[0]=="right":
            action=actions[1][2]
        elif max_list[0]=="up_left":
            action=actions[0][0]
        elif max_list[0]=="up_right":
            action=actions[0][2]
        elif max_list[0]=="down_left":
            action=actions[2][0]
        elif max_list[0]=="down_right":
            action=actions[2][2]
        else:
            action=random.randint(1, 9)
    act=action        
    return act

#Receives a grid zone and count fire cells
def Count_Burned_Trees(env,zone):
    counter=0
    for row in range(zone.shape[0]):
        for col in range(zone.shape[1]):
            if zone[row][col]==env.fire:
                counter+=1
    return counter

#Receives a grid zone and calculate fire coefficient
def Calculate_Fire_Coefficient(env,zone):    
    coefficient=0
    for row in range(zone.shape[0]):
        for col in range(zone.shape[1]):
            if zone[row][col]==env.fire:
                coefficient+=1
            if zone[row][col]==env.tree:
                coefficient+=env.p_fire
            if zone[row][col]==env.empty:
                coefficient-=env.p_tree
    coefficient=coefficient/(zone.shape[0]*zone.shape[1])    
    return coefficient

#Get neighborhood of agent according to vision range
def get_neighborhood(grid,pos,vision):
    pos_row=pos[0]
    pos_col=pos[1]    
    neighborhood=grid[pos_row:pos_row+1+vision*2,pos_col:pos_col+1+vision*2]
    return neighborhood

def ExpandGrid(grid,vision):        
        size = grid.shape        
        PadGrid = np.zeros((size[0],size[1]), dtype=np.int16)        
        for i in range(size[0]):
            for j in range(size[1]):
                if(grid[i][j][0]==1):
                    PadGrid[i][j]=0
                elif(grid[i][j][1]==1):
                    PadGrid[i][j]=1
                else:
                    PadGrid[i][j]=2
        size=PadGrid.shape
        PadGrid2 = np.zeros((size[0]+2*vision,size[1]+2*vision), dtype=np.int16)
        PadGrid2[vision:-vision,vision:-vision] = PadGrid
        return PadGrid2

def DUMB_ONE(args):
    assert isinstance(args, dict), "This function requieres a dict of kwards to perform. Please provide one."
    expected_args = ['env']
    for arg in expected_args:
        if args.get(arg) is None:
            raise "At least the argument {} is missing! Please do pass correct arguments".format(arg)
    env = args['env']
    actions = env.action_set
    return actions[0]

def potential_one(args):
    assert isinstance(args, dict), "This function requieres a dict of kwards to perform. Please provide one."
    expected_args = ['env','observation']
    for arg in expected_args:
        if args.get(arg) is None:
            raise "At least the argument {} is missing! Please do pass correct arguments".format(arg)
    env = args['env']
    grid, pos, remain_steps = args['observation']

    return None