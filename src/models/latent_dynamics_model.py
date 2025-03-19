import torch
import torch.nn as nn

from typing import Tuple

class HashedTensor():
    def __init__(self, tensor):
        self.tensor = tensor.clone().detach().cpu()

    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)

class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        delta_mode: bool = False,
    ):
        '''Initialize the LatentDynamicsModel which 
        is a neural network that predicts the next state, the
        reward, and if the episode is terminated given the 
        current state and action. 

        Args:
            state_dim (int): Number of dimensions in the state vector.
            action_dim (int): Number of dimensions in the action vector. Since 
            actions are one-hot encoded, this is the magnitude of the action space.
            delta_mode (bool, optional): If True, state predictions will be additive 
            instead of absolute. Defaults to False.
        '''
        super(LatentDynamicsModel, self).__init__()
        self.delta_mode = delta_mode
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.latent_out = nn.Linear(32, state_dim)
        self.reward_out = nn.Linear(32, 1)
        self.term_out = nn.Linear(32, 1)
        self.delta_mode = delta_mode
        self.memoize_on = False
        self.memos = {}
        
    def start_memoize(self):
        self.memoize_on = True

    def stop_memoize(self):
        self.memoize_on = False

    def set_memo(
        self, 
        latent_obs: torch.Tensor, 
        action: torch.Tensor, 
        latent_next_obs: torch.Tensor, 
        reward: torch.Tensor, 
        term: torch.Tensor
    ):
        '''Memoize the prediction of the latent dynamics model.

        Args:
            latent_obs (torch.Tensor): The latent observation before the action.
            action (torch.Tensor): The one-hot encoded action.
            latent_next_obs (torch.Tensor): The predicted latent observation after the action.
            reward (torch.Tensor): The predicted reward received after the action.
            term (torch.Tensor): The predicted termination signal after the action.
        '''
        memo_key = HashedTensor(torch.cat([latent_obs, action], dim=-1))
        self.memos[memo_key] = (latent_next_obs, reward, term)

    def clear_memos(self):
        self.memos = {}
        
    def forward(
        self, 
        latent_obs: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        '''Run the latent dynamics model on the given latent observation and action.

        Args:
            latent_obs (torch.Tensor): The latent observation before the action.
            action (torch.Tensor): The one-hot encoded action.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]: predicted latent observation,
            predicted reward, predicted termination signal, and a boolean indicating if memoization
            was used to retrieve the prediction.
        '''
        x = torch.cat([latent_obs, action], dim=-1)
        # check if this input has been memoized
        if self.memoize_on:   # We will assume no batches during memoization.
            memo_key = HashedTensor(x)
            if memo_key in self.memos:
                return *self.memos[memo_key], True
        
        # run inputs through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # extract the predictions
        if self.delta_mode:
            pred_latent_next_obs = latent_obs + self.latent_out(x)
        else:
            pred_latent_next_obs = self.latent_out(x)
        pred_reward = self.reward_out(x)
        pred_terminate = torch.sigmoid(self.term_out(x))
        
        # add to memoization
        if self.memoize_on:
            self.memos[memo_key] = (pred_latent_next_obs, pred_reward, pred_terminate)
            
        return pred_latent_next_obs, pred_reward, pred_terminate, False
    
    def run(self, latent_obs, action):
        return self(latent_obs, action)