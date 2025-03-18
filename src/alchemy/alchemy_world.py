import numpy as np
from typing import List

from .potion import Potion

class StateActionPair():
    def __init__(self, state: np.ndarray, action: int):
        self.state = state
        self.action = action
        
    def __iter__(self):
        return iter((self.state, self.action))  

class AlchemyWorld():
    '''This class stores the transition dynamics and blocked actions for 
    a given alchemy episode. The actions are the indices of the actions 
    list. The transition dynamics are the contents of the actions. The 
    blocked_pairs are the blocked (state, action) pairs. The class is 
    functionally a data wrapper.
    '''
    def __init__(
        self,
        actions: List[Potion],
        blocked_pairs: List[StateActionPair],
    ):
        '''Initialize an alchemy world.

        Args:
            actions (list[Potion]): List of potions that the alchemist can use.
            blocked_pairs (list[StateActionPair]): List of state-action pairs that are blocked.
        '''
        self.actions = actions
        self.blocked_pairs = blocked_pairs
        
    def add_blocked_pair(self, state: np.ndarray, action: int):
        '''Add a blocked state-action pair to the world.

        Args:
            state (np.ndarray): The state of the rock.
            action (int): The action to be blocked.
        '''
        self.blocked_pairs.append(StateActionPair(state, action))