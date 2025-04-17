import pygame
import random
from Environment import *

env = Environment(state=state)

class RandomAgent:
    def __init__(self, env) -> None:
        self.env = env

    def get_Action(self, state, events=None, step=None):
        # Action: left = 1; right=2; rotate=3; no_action = 0; illegal = none
        if step % 15 != 0:
            return None
        action = random.randint(0, 3)
        # if env.not_legal(state, action):
        #     return None
        return action
    
    def get_end_Action(self, events=None):
        action = 6
        return action