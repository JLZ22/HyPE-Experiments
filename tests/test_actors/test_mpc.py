import gymnasium as gym
import torch 
import numpy as np 

from src.actors import MPC_Actor
from src.models import LatentDynamicsModel, LatentEncoder

NUM_FEATURES = 6
LATENT_SIZE = 3

def init_vars():
    action_space = gym.spaces.Discrete(NUM_FEATURES+1)
    actor = MPC_Actor(action_space)
    model = LatentDynamicsModel(LATENT_SIZE, action_space.n)
    enc = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
    return actor, model, enc

class TestMPC_Actor():
    @torch.no_grad()
    def test_dfs(self):
        actor, model, enc = init_vars()
        obs = torch.randn(1, NUM_FEATURES, dtype=torch.float)
        import logging
        logging.info(obs)
        latent_obs, _ = enc(obs)
        depth = 3
        reward, path = actor._dfs(model, latent_obs, depth, 0, [])
        assert len(path) <= depth, 'Path should have length less than or equal to depth.'
        assert isinstance(reward.item(), float), 'Reward should be a float.'
        for p in path:
            assert isinstance(p, int), 'Path should be a list of integers.'
            assert p in range(NUM_FEATURES+1), 'Path should be a list of valid actions.'
    
    @torch.no_grad()
    def test_steps(self):
        actor, model, enc = init_vars()
        enc = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = np.random.random(NUM_FEATURES)
        next_action = actor.step(
            model,
            enc,
            obs,
            3
        )
        assert isinstance(next_action, int), 'Next action should be an integer.'
        assert next_action in range(NUM_FEATURES+1), 'Next action should be a valid action.'