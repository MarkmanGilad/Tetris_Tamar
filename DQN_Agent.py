import torch
import random
import math
# from DQN_Conv import DQN
from DQN import DQN
from State import *

epsilon_start = 1
epsilon_final = 0.1
epsilon_decay = 300

class DQN_Agent:
    def __init__(self, parametes_path = None, train = False, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train = train
        self.env = env
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def direction(self, col, new_col):
        if new_col > col:
            return 2 # right
        elif new_col < col:
            return 1 # left
        return 0 # stay
        

    def get_Action (self, state, epoch = 0, events= None, train = False) -> tuple:
        # state = state.toTensor()
        _, col, _ = state.falling_piece
        _, states_dqn, actions, cleared_rows = self.env.all_states(state) 
        
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                idx = random.randrange(len(actions))
                rotate, new_col = actions[idx]
                cleared_row = cleared_rows[idx]
                if rotate == 0:
                    return self.direction(col, new_col), cleared_row
                return 3, cleared_row # rotate
                        
        
        # Convert to a NumPy array first
        np_batch = np.stack(states_dqn)  # shape: [actions, 10]

        # Then convert to PyTorch tensor
        tensor_batch = torch.from_numpy(np_batch).to(dtype=torch.float32)

        with torch.no_grad():
            Q_values = self.DQN(tensor_batch)
        max_index = torch.argmax(Q_values)
        rotate, new_col = actions[max_index]
        cleared_row = cleared_rows[max_index]
        if rotate == 0:
            return self.direction(col, new_col), cleared_row
        return 3, cleared_row # rotate

    def get_Actions_Values (self, states):
        with torch.no_grad():
            Q_values = self.DQN(states)
            max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        
        return max_indices.reshape(-1,1), max_values.reshape(-1,1)

    def Q (self, states, actions):
        Q_values = self.DQN(states)
        rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
        cols = actions.reshape(-1,1)
        return Q_values[rows, cols]

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        if epoch < decay:
            return start - (start - final) * epoch/decay
        return final
        
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)
    
    def Q (self, states, actions):
        Q_values = self.DQN(states) 
        rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
        cols = actions.reshape(-1,1).to(torch.int)
        return Q_values[rows, cols]

    def get_end_Action(self, events=None):
        action = 6
        return action