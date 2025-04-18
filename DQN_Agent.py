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

        next_states, next_states_dqn, actions, cleared_rows = self.env.all_states(state) 
        
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                idx = random.randrange(len(actions))
                return next_states[idx], next_states_dqn[idx], actions[idx], cleared_rows[idx]
                
        # Convert to a NumPy array first
        np_batch = np.stack(next_states_dqn)  # shape: [actions, 10]

        # Then convert to PyTorch tensor
        tensor_batch = torch.from_numpy(np_batch).to(dtype=torch.float32)

        with torch.no_grad():
            Q_values = self.DQN(tensor_batch)
        max_index = torch.argmax(Q_values)
        return next_states[max_index], next_states_dqn[max_index], actions[max_index], cleared_rows[max_index]

    def get_next_states_Values (self, states):
        
        states_dqn_lst, next_Q_values_lst = [], []
        for state in states:
            next_states, next_states_dqn, actions, cleared_rows = self.env.all_states(state) 
            # Convert to a NumPy array first
            np_batch = np.stack(next_states_dqn)  # shape: [actions, 10]

            # Then convert to PyTorch tensor
            tensor_batch = torch.from_numpy(np_batch).to(dtype=torch.float32)

            with torch.no_grad():
                Q_values = self.DQN(tensor_batch)
            max_index = torch.argmax(Q_values)
            states_dqn_lst.append(next_states_dqn[max_index])
            next_Q_values_lst.append(Q_values[max_index])

        states_dqn_tensor = torch.from_numpy(np.stack(states_dqn_lst)).to(torch.float32)
        next_Q_values_tensor = torch.vstack(next_Q_values_lst)
        return states_dqn_tensor, next_Q_values_tensor 

    def Q (self, states_dqn):
        return self.DQN(states_dqn)
        
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
        

    def get_end_Action(self, events=None):
        action = 6
        return action