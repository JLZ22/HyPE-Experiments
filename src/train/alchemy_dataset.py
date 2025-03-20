import numpy as np 
import torch

from typing import List
from torch.utils.data import Dataset

class AlchemyDataset(Dataset):
    def __init__(self):
        # x
        self.observations = None
        self.actions = None
        
        # y
        self.next_observations = None
        self.rewards = None
        self.terms = None
        self.states = []
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx],
            'next_observation': self.next_observations[idx],
            'reward': self.rewards[idx],
            'term': self.terms[idx],
            'state': torch.tensor(self.states[idx], dtype=torch.float)
        }
        
    def add_transition(
        self,
        observation: List[int | float],
        action: List[int | float],
        next_observation: List[int | float],
        reward: float,
        term: int,
        state: np.ndarray,
    ):
        '''Add a transition to the dataset. 

        Args:
            observation (List[int | float]): The observation before the action.
            action (List[int | float]): The one-hot encoded action.
            next_observation (List[int | float]): The observation after the action.
            reward (float): The reward received after the action.
            term (int): If the episode is terminated after the action.
            state (np.ndarray): The state of the environment before the action.
        '''
        self.states.append(state.copy())
        if self.observations is not None:
            concat = lambda existing_points, new_point: torch.cat([
                existing_points, 
                torch.tensor(
                    np.expand_dims(new_point, axis=0),
                    dtype=torch.float16
                )
            ])
            self.observations       = concat(self.observations, observation)
            self.actions            = concat(self.actions, action)
            self.next_observations  = concat(self.next_observations, next_observation)
            self.rewards            = concat(self.rewards, np.array([reward]))
            self.terms              = concat(self.terms, np.array([term]))
        else:
            make_tensor = lambda point: torch.tensor(
                np.expand_dims(point, axis=0), 
                dtype=torch.float16
            )
            self.observations       = make_tensor(observation)
            self.actions            = make_tensor(action)
            self.next_observations  = make_tensor(next_observation)
            self.rewards            = make_tensor(np.array([reward]))
            self.terms              = make_tensor(np.array([term]))