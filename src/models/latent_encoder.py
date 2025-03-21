import torch 
import torch.nn as nn

from typing import Tuple
from src.utils import HashedTensor, get_device

class LatentEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        device: torch.device = None,
    ):
        '''Encode the state of the environment into a latent space.

        Args:
            state_dim (int): The number of dimensions in the state vector.
            latent_dim (int): The target number of dimensions in the latent space.
            device (torch.device, optional): The device to run the model on. 
            If None, the best compatible one will be chosen for you. Defaults to None.
        '''
        super(LatentEncoder, self).__init__()
        self.device = get_device(device)
        self.fc1 = nn.Linear(state_dim, 256).to(self.device)
        self.fc2 = nn.Linear(256, latent_dim).to(self.device)
        self.memoize_on = False
        self.memos = {}
    
    def start_memoize(self):
        self.memoize_on = True
    
    def stop_memoize(self):
        self.memoize_on = False
        
    def set_memo(self, state: torch.Tensor):
        '''Memoize the prediction of the latent encoder.

        Args:
            state (torch.Tensor): The state of the environment.
        '''
        memo_key = HashedTensor(state)
        self.memos[memo_key] = state
        
    def clear_memos(self):
        self.memos = {}
        
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        '''Run the latent encoder on the state.

        Args:
            state (torch.Tensor): The state of the environment.

        Returns:
            Tuple[torch.Tensor, bool]: The latent representation of the state 
            and if the prediction was memoized.
        '''
        state = state.to(self.device)
        if self.memoize_on:
            memo_key = HashedTensor(state)
            if memo_key in self.memos:
                return self.memos[memo_key], True
        
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        if self.memoize_on:
            self.memos[memo_key] = x
        
        return x, False
