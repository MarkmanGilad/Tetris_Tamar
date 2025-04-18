import numpy as np

from Environment import Environment
from State import State

def all_states():
    state = State()
    env = Environment(state=state)
    next_states, next_states_dqn, actions, cleared_rows = env.all_states(state)

    print(next_states)
    print(next_states_dqn)
    print(actions)
    print(cleared_rows)
    

def pieces_test():
    pieces = {
            1:np.array([[1, 1, 1, 1]]),
            2:np.array([[2, 0, 0],[2, 2, 2]]),
            3:np.array([[0, 0, 3],[3, 3, 3]]),
            4:np.array([[4, 4],[4, 4]]),
            5:np.array([[0, 5, 5],[5, 5, 0]]),
            6:np.array([[0, 6, 0],[6, 6, 6]]),
            7:np.array([[7, 7, 0],[0, 7, 7]])
        }
    for i in range(4):
        print(np.rot90(pieces[3], k=i) )

def height() :
    board = np.array([
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]])
    
    
    
    return 
# pieces_test()
all_states()
