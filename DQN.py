import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


input_size = 7 # holes, landing height, wells, bumpiness, total height, full rows, done

gamma = 0.95 
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, 31)
        self.linear2 = nn.Linear(32, 64)
        self.output = nn.Linear(64, 1)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- dones)
        return self.MSELoss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

   
    def __call__(self, states):
        return self.forward(states)