import pytest
import torch 

from src.models.latent_dynamics_model import LatentDynamicsModel

LATENT_SIZE = 3
NUM_ACTIONS = 9

class TestLatentDynamicsModel():
    def test_forward(self):
        model = LatentDynamicsModel(LATENT_SIZE, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, LATENT_SIZE)
        sample_action = torch.zeros(1, NUM_ACTIONS)
        latent_obs, reward, term, _ = model.forward(sample_state, sample_action)
        assert latent_obs.shape == sample_state.shape, 'Latent observation shape should match input shape.'
        assert reward.shape == (1, 1), 'Reward shape should be (1, 1).'
        assert term.shape == (1, 1), 'Termination signal shape should be (1, 1).'
    
    def test_memo(self):
        model = LatentDynamicsModel(LATENT_SIZE, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, LATENT_SIZE)
        sample_action = torch.zeros(1, NUM_ACTIONS)
        
        model.start_memoize()
        _, _, _, did_mem = model.forward(sample_state, sample_action)
        assert not did_mem, 'Model should not memoize on first forward pass.'
        _, _, _, did_mem = model.forward(sample_state, sample_action)
        assert did_mem, 'Model should memoize on second forward pass.'
        model.clear_memos()
        _, _, _, did_mem = model.forward(sample_state, sample_action)
        assert not did_mem, 'Model should not memoize after clearing memos.'
        model.clear_memos()
        model.set_memo(
            sample_state, 
            sample_action, 
            sample_state, 
            torch.tensor([0.0]), 
            torch.tensor([0.0])
        )
        _, _, _, did_mem = model.forward(sample_state, sample_action)
        assert did_mem, 'Model should memoize after setting memo.'
        model.stop_memoize()
        _, _, _, did_mem = model.forward(sample_state, sample_action)
        assert not did_mem, 'Model should not memoize after stopping memoization.'
        
    def test_failure(self):
        model = LatentDynamicsModel(LATENT_SIZE, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, LATENT_SIZE + 1)
        sample_action = torch.zeros(1, NUM_ACTIONS + 1)
        with pytest.raises(RuntimeError):
            model.forward(sample_state, sample_action)
            
    def test_device_consistency(self):
        model = LatentDynamicsModel(LATENT_SIZE, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, LATENT_SIZE).to('cpu')
        sample_action = torch.zeros(1, NUM_ACTIONS).to('cpu')
        _, _, _, _ = model.forward(sample_state, sample_action)