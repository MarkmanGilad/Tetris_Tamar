import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
# input_size = 220 # board (10 * 20) + falling piece (1 + 1 + 4 * 4) + fall_speed (1) + next_piece (1) = 220
input_size = 7 # holes, landing height, wells, bumpiness, total height, full rows, done
# input_size = 6 # holes, landing height, wells, bumpiness, total height, full rows


layer1 = 128
layer2 = 256
layer3 = 64
output_size = 4 # Q(state)-> 4 values of stay, left, right, rotate
gamma = 0.95 
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, layer3)
        self.output = nn.Linear(layer3, output_size)
        # self.softmax = nn.Softmax(dim=-1)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        # x = self.softmax(x)
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