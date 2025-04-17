import pygame
from Environment import *

env = Environment(state=state)

class HumanAgent:
    def __init__(self, env) -> None:
        self.env = env

    def get_Action(self, state, events, step=None):
        # Action: left = 1; right=2; rotate=3; down=4; not-down=5; no_action = None; illegal = none
        action = None
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    # הזזה שמאלה
                    action  = 1
                elif event.key == pygame.K_RIGHT:
                    # הזזה ימינה
                    action = 2
                elif event.key == pygame.K_UP:
                    # סיבוב
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 4
                # if env.not_legal(state, action):
                #     return None
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    action = 5
        return action
    
    def get_end_Action(self, events):
        action = None
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # הזזה שמאלה
                    action  = 6
        return action