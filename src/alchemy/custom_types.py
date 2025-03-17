import numpy as np

from .potion import Potion

# a (state, action) pair 
type StateActionPair = tuple[np.ndarray, int]

# A world is a list of potions and a list of blocked (state, action) pairs
type World = tuple[list[Potion], list[StateActionPair]]