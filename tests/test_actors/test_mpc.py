import gymnasium as gym
import torch 

from src.actors import MPC_Actor
from src.models import LatentDynamicsModel, LatentEncoder
from src.utils import get_device

NUM_FEATURES = 6
LATENT_SIZE = 3

class TestMPC_Actor():
    def test_dfs(self):
        action_space = gym.spaces.Discrete(NUM_FEATURES+1)
        actor = MPC_Actor(action_space)
        model = LatentDynamicsModel(LATENT_SIZE, action_space.n)
        enc = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = torch.randn(1, NUM_FEATURES)
        latent_obs, _ = enc(obs)
        depth = 3
        reward, path = actor._dfs(model, latent_obs, depth, 0, [])
        assert len(path) <= depth, 'Path should have length less than or equal to depth.'
        assert isinstance(reward.item(), float), 'Reward should be a float.'
        for p in path:
            assert isinstance(p, int), 'Path should be a list of integers.'
            assert p in range(NUM_FEATURES+1), 'Path should be a list of valid actions.'