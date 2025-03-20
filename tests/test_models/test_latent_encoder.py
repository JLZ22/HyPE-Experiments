import pytest
import torch

from src.models import LatentEncoder

NUM_FEATURES = 6
LATENT_SIZE = 3

class TestLatentEncoder():
    def test_forward(self):
        model = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = torch.randn(1, NUM_FEATURES)
        latent_obs, _ = model(obs)
        assert latent_obs.shape == (1, LATENT_SIZE), 'Latent observation shape is incorrect.'
        
    def test_failure(self):
        model = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = torch.randn(1, NUM_FEATURES + 1)
        with pytest.raises(RuntimeError):
            model(obs)
            
    def test_memo(self):
        model = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = torch.randn(1, NUM_FEATURES)
        
        model.start_memoize()
        _, did_mem = model(obs)
        assert not did_mem, 'Model should not have retrived a memoized tensor on first forward pass.'
        _, did_mem = model(obs)
        assert did_mem, 'Model should have retrived the previous tensor on this pass.'
        model.clear_memos()
        
        _, did_mem = model(obs)
        assert not did_mem, 'Model should not have retrived a memoized tensor. Memos were cleared and should be empty.'
        model.set_memo(obs)
        _, did_mem = model(obs)
        assert did_mem, 'Model should have retrived the set tensor on this pass.'
        model.stop_memoize()
        
        _, did_mem = model(obs)
        assert not did_mem, 'Model should not have retrived a memoized tensor. Memoization was stopped.'
        
    def test_device_consistency(self):
        model = LatentEncoder(NUM_FEATURES, LATENT_SIZE)
        obs = torch.randn(1, NUM_FEATURES).to('cpu')
        _, _ = model(obs)
        obs = torch.randn(1, NUM_FEATURES).to('cpu')
        _, _ = model(obs)