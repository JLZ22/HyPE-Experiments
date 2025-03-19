import torch 

from src.models.latent_dynamics_model import LatentDynamicsModel

NUM_STATES = 10
NUM_ACTIONS = NUM_STATES + 1

class TestLatentDynamicsModel():
    def test_forward(self):
        model = LatentDynamicsModel(NUM_STATES, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, NUM_STATES)
        sample_action = torch.zeros(1, NUM_ACTIONS)
        latent_obs, reward, term, _ = model.forward(sample_state, sample_action)
        assert latent_obs.shape == sample_state.shape, 'Latent observation shape should match input shape.'
        assert reward.shape == (1, 1), 'Reward shape should be (1, 1).'
        assert term.shape == (1, 1), 'Termination signal shape should be (1, 1).'
    
    def test_memo(self):
        model = LatentDynamicsModel(NUM_STATES, NUM_ACTIONS, delta_mode=True)
        sample_state = torch.randn(1, NUM_STATES)
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