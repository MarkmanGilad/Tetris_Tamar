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

    def push (self, state , action, reward, next_state, next_state_dqn, done):
        self.buffer.append((state, action, reward, next_state, next_state_dqn, done))
        # self.buffer.append((state.toTensor(), action.type(torch.float32), torch.tensor(reward, dtype=torch.float32), next_state.toTensor(), torch.tensor(done, dtype=int)))

    
    def sample (self, batch_size):
        states, actions, rewards, next_states, next_states_dqn, dones = zip(*random.sample(self.buffer, batch_size))
        # states = torch.vstack(state_tensors)
        # actions= torch.vstack(action_tensor)
        np_batch = np.stack(rewards)
        rewards_tensor = torch.from_numpy(np_batch).to(torch.float32).reshape(-1,1)
        # next_states = torch.vstack(next_state_tensors)
        np_batch = np.stack(next_states_dqn)
        next_states_dqn_tensor = torch.from_numpy(np_batch).to(torch.float32)
        np_batch = np.stack(dones)
        dones_tensor = torch.from_numpy(np_batch).to(torch.float32).reshape(-1,1)
        return states, actions, rewards_tensor, next_states, next_states_dqn_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)
