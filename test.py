import numpy as np

from Environment import Environment
from State import State

state = State()
env = Environment(state=state)
env.select_falling_piece(state)
states, actions = env.all_states(state)

print(states)
print(actions)