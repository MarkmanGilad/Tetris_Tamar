from collections import deque
import random
import torch
import numpy as np

capacity = 500000

class ReplayBuffer:
    def __init__(self, capacity= capacity, path = None) -> None:
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)

    def push (self, state , action, reward, next_state, done):
        self.buffer.append((torch.from_numpy(state), torch.tensor(action, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), torch.from_numpy(next_state), torch.tensor(done, dtype=int)))
        # self.buffer.append((state.toTensor(), action.type(torch.float32), torch.tensor(reward, dtype=torch.float32), next_state.toTensor(), torch.tensor(done, dtype=int)))

    
    def sample (self, batch_size):
        if (batch_size > self.__len__()):
            batch_size = self.__len__()
        state_tensors, action_tensor, reward_tensors, next_state_tensors, dones_tensor = zip(*random.sample(self.buffer, batch_size))
        states = torch.vstack(state_tensors)
        actions= torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        dones = torch.vstack(dones_tensor)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
