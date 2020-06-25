import helicopter
import heuristic
import rollout

# -----------------------Preparing the experiment----------------------------
# Environment parameters
N_ROW = 16              #Grid Rows
N_COL = 16              #Grid Columns
Init_Row=7              #Helicopter initial row
Init_Col=7              #Helicopter initial col
P_FIRE = 0.02           #Probability to turn a green cell into ared cell
P_TREE = 0.08           #Probability to turn an empty cell into a green cell
FREEZE = 4              #Movements of Helicopter after update Automata
# Symbols for cells
TREE = 0
FIRE = 2
EMPTY = 1
## Environment cost shape
C_TYPE = 'custom'
C_TREE = -5.0           # Costs asociated with its type cell
C_FIRE = 1.0            #
C_EMPTY = 0.0           #
C_HIT = -1.5            # Asociated to put down a fire.
C_STEP = 0.0            # Per step given on the environment
C_MOVE = 2.0            # Cost to chage position
# Experiment parameters
N_TRAIN = 15
N_STEPS = 50
N_SAMPLES = 30
K_Rollout = 10 * FREEZE
LOOKAHEAD = 1

if __name__ == '__main__':
    env = helicopter.EnvMakerForestFire(
        n_row = N_ROW, n_col = N_COL, 
        p_tree = P_TREE, p_fire = P_FIRE,
        init_pos_row = Init_Row, init_pos_col = Init_Col, 
        moves_before_updating = FREEZE,  
        tree = TREE, empty = EMPTY, fire = FIRE,
        reward_type = C_TYPE, reward_tree = C_TREE, reward_fire = C_FIRE,
        reward_empty = C_EMPTY, reward_hit = C_HIT, reward_step = C_STEP,
        reward_move = C_MOVE)
    H = heuristic.Heuristic_m2_v1
    exp = rollout.Experiment(env, H, H_mode = 21,
        N_TRAIN=N_TRAIN, N_STEPS=N_STEPS,N_SAMPLES=N_SAMPLES, 
        K=K_Rollout, LOOKAHEAD=LOOKAHEAD, MIN_OBJECTIVE=True, 
        ALPHA=0.9, EPSILON=0.0, RUN_GIF=True)
    exp.run()
    #exp.make_gif(RUN=True)
    exp.LOOKAHEAD = 2
    exp.run()
    exp.LOOKAHEAD = 3
    exp.run()
    #exp.make_gif(RUN=True,fps=7)
    #exp.policy_test(N_TEST=5,N_STEPS=N_STEPS)
    #exp.make_gif(TEST=True,fps=7)
     