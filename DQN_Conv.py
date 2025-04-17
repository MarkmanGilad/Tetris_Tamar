import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 220 # board (10 * 20) + falling piece (1 + 1 + 4 * 4) + fall_speed (1) + next_piece (1) = 220

layer1 = 512
layer2 = 256
layer3 = 64
output_size = 4 # Q(state)-> 4 values of stay, left, right, rotate
gamma = 0.99                                                                # gilad
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.board_height=20
        self.board_width=10 
        self.piece_height=4
        self.piece_width=4 
        self.board_size = self.board_height * self.board_width      # 200
        self.piece_start = self.board_size + 1                      # 201
        self.piece_size = self.piece_height * self.piece_width + 2     # 18
        conv_output_size = 32 * 16 * 6  # = 3072

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        # Output: (batch, 16, 18, 8)  because: 20-3+1 = 18 and 10-3+1 = 8.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        # Output: (batch, 32, 16, 6)  since: 18-3+1 = 16 and 8-3+1 = 6.
        conv_output_size = 32 * 16 * 6  # = 3072
        # Combined feature size from board and falling piece.
        fc_input_size = conv_output_size + self.piece_size


        self.linear1 = nn.Linear(fc_input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, layer3)
        self.output = nn.Linear(layer3, output_size)
        
        
        self.MSELoss = nn.MSELoss()

        

    def forward (self, x):
        board = x[:, :self.board_height * self.board_width]  # shape: (batch, 200)
        piece = x[:, self.piece_start: self.piece_start + self.piece_size ]    # shape: (batch, 16)
        board = board.view(-1, 1, self.board_height, self.board_width)  # (batch, 1, 20, 10)

        board = F.relu(self.conv1(board))
        board = F.relu(self.conv2(board))
        board = board.view(board.size(0), -1)  # Flatten to (batch, 3072)

        # Process the falling piece: simply flatten.
        piece = piece.view(piece.size(0), -1)  # (batch, 18)
        
        # Concatenate the board and piece features.
        x = torch.cat((board, piece), dim=1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
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