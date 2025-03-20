import gymnasium as gym
import numpy as np
import torch

from typing import List, Tuple
from src.utils import get_device

class MPC_Actor():
    def __init__(
        self, 
        action_space: gym.spaces.Discrete,
    ):
        '''Initialize a DFS Actor who traverses 
        a model of the environment using DFS to 
        find the best action. 

        Args:
            action_space (gym.spaces.Discrete): The action space of the environment.
        '''
        self.action_space = action_space
        self.path = []
        
    def _dfs(
        self,
        model: torch.nn.Module,
        latent_obs: torch.Tensor,
        depth: int,
        current_reward: float,
        path: List[int],
    ) -> Tuple[float, List[int]]: 
        '''Depth-Limited search through the latent dynamics model 
        to find the best sequence of actions and reward. 

        Args:
            model (torch.nn.Module): Model of the environment.
            latent_obs (torch.Tensor): Current observation of the environment in latent space.
            depth (int): The depth limit of the search.
            current_reward (float): The current reward of the path.
            path (List[int]): The current path through the model.
            
        Returns:
            Tuple[float, List[int]]: The best reward and path found.
        '''
        if depth == 0:
            return current_reward, path
        best_reward = -float('inf')
        best_path = None
        for action in reversed(range(self.action_space.n)):
            # One-hot encode the action
            action_one_hot = torch.zeros(self.action_space.n)
            action_one_hot[action] = 1
            action_one_hot = action_one_hot.unsqueeze(0)
            
            # Predict the next latent observation, reward, and termination signal
            next_latent_obs, pred_reward, pred_term, _ = model.run(latent_obs, action_one_hot)
            
            # Calculate the cumulative reward and keep track of the path we took
            cumulative_reward = current_reward + pred_reward
            potential_path = path + [action]
            
            # check if we have finished and if we've found a better path
            if pred_term >= 0.5 and cumulative_reward > best_reward:
                    best_reward = cumulative_reward
                    best_path = potential_path
                    
            # Recurse to the next depth
            ret_reward, ret_path = self._dfs(
                model, 
                next_latent_obs, 
                depth - 1, 
                cumulative_reward, 
                potential_path
            )
            
            # If we've found a better path, update the best path
            if ret_reward > best_reward:
                best_reward = ret_reward
                best_path = ret_path
        return best_reward, best_path
    
    def step(
        self,
        model: torch.nn.Module,
        enc: torch.nn.Module,
        obs: np.ndarray,
        horizon: int=4,
    ) -> int: 
        '''Perform depth limited search to find the action 
        that leads to the path that maximizes the reward.

        Args:
            model (torch.nn.Module): The model of the world.
            enc (torch.nn.Module): The encoder that encodes the observation into latent space.
            obs (np.ndarray): The raw observation.
            horizon (int, optional): The horizon of the search. Defaults to 4.

        Returns:
            int: The action.
        '''
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        import logging
        logging.info(f'obs: {obs}')
        latent_obs, _ = enc(obs)
        _, best_path = self._dfs(model, latent_obs, horizon, 0, [])
        return best_path[0]