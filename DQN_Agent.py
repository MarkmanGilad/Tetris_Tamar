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
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state, epoch = 0, events= None, train = False) -> tuple:
        # state = state.toTensor()
        actions = [0,1,2,3]
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        with torch.no_grad():
            Q_values = self.DQN(state)
        return torch.argmax(Q_values)

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