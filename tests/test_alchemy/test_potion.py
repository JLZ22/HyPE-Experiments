import numpy as np
import pytest
import scipy.stats as stats

from src.alchemy.potion import Potion

class TestPotion():        
    def test_out_of_range(self):
        '''Test that the distribution is not in the range [0, 1].
        '''
        with pytest.raises(ValueError):
            Potion(0, 'luster', stats.norm(-1, 1))
            
    def test_distributions(self):
        '''Test the Potion class with different distributions.
        '''
        def truncnorm(mean, std, lower, upper):
            a, b = (lower - mean) / std, (upper - mean) / std
            return stats.truncnorm(a, b, loc=mean, scale=std)
        
        distributions = [
            stats.binom(1, 0.5),
            truncnorm(0.5, 0.2, 0, 1),
            stats.uniform(0, 1),
            stats.truncexpon(1)
        ]
        
        for dist in distributions:
            Potion(0, 'luster', dist)
            
    def test_feature_idx_out_of_range(self):
        '''Test that the Potion class will raise 
        errors appropriately when the feature index 
        is out of range or invalid. 
        '''
        with pytest.raises(AssertionError):
            Potion(-1, 'luster', stats.binom(1, 1))
        with pytest.raises(IndexError):
            p = Potion(10, 'luster', stats.binom(1, 1))
            p.use_on(np.array([0]))