import pytest
import scipy.stats as stats

from src.alchemy.potion import Potion

class TestPotion():        
    def test_out_of_range(self):
        with pytest.raises(ValueError):
            Potion(0, 'luster', stats.norm(-1, 1))
            
    def test_distributions(self):
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