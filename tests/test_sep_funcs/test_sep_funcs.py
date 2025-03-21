import gymnasium as gym
import torch 

from src.sep_funcs import *
from src.models import LatentDynamicsModel

ACTION_SPACE = gym.spaces.Discrete(6)
NUM_FEATURES = 5
LATENT_SIZE = 3

def test_run_actions():
    model = LatentDynamicsModel(LATENT_SIZE, ACTION_SPACE.n)
    action_seq = [0, 1, 2, 5, 4, 5]
    curr_state = torch.randn(1, LATENT_SIZE)
    state_seq, total_reward, num_non_term_states = run_actions(model, action_seq, curr_state, ACTION_SPACE)
    assert len(state_seq) == len(action_seq) + 1
    for state in state_seq:
        assert state.shape == (1, LATENT_SIZE), "Shape of state is incorrect."
    assert isinstance(total_reward, float), "Total reward should be a float."
    assert isinstance(num_non_term_states, int), "Number of non-terminal states should be an int."
    assert num_non_term_states >= 0, "Number of non-terminal states should be non-negative."
    
def test_l2across():
    models = [LatentDynamicsModel(LATENT_SIZE, ACTION_SPACE.n) for _ in range(3)]
    action_seq = [0, 1, 2, 5, 4, 5]
    curr_state = torch.randn(1, LATENT_SIZE)
    l2_dist = sepfunc_l2across(models, action_seq, curr_state, ACTION_SPACE)
    assert isinstance(l2_dist, float), "L2-distance should be a float."
    assert l2_dist >= 0, "L2-distance should be non-negative."

def test_central_deviation():
    models = [LatentDynamicsModel(LATENT_SIZE, ACTION_SPACE.n) for _ in range(3)]
    action_seq = [0, 1, 2, 5, 4, 5]
    curr_state = torch.randn(1, LATENT_SIZE)
    central_dev = sepfunc_central_deviation(models, action_seq, curr_state, ACTION_SPACE)
    assert isinstance(central_dev, float), "Central deviation should be a float."
    assert central_dev >= 0, "Central deviation should be non-negative."