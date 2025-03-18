import numpy as np
import scipy.stats as stats
from src.alchemy.continuous_stochastic_alchemy_env import ContinuousStochasticAlchemyEnv

class TestContinuousStochasticAlchemyEnv():
    def test_generate_world(self):
        def truncnorm(mean, std, lower, upper):
            a, b = (lower - mean) / std, (upper - mean) / std
            return stats.truncnorm(a, b, loc=mean, scale=std)
        
        distributions = [
            truncnorm(0.4, 0.3, 0, 1),
            truncnorm(0.5, 0.2, 0, 1),
            stats.truncexpon(1)
        ]
        env = ContinuousStochasticAlchemyEnv()
        env.reset(env.generate_world_from_distributions(distributions))
        assert len(env.world.actions) == env.num_features + 1, 'The number of actions is incorrect.'
        assert len(env.world.blocked_pairs) <= env.max_blocks, 'Too many blocked pairs.'
        for pot in env.world.actions[:-1]:
            assert pot.feature_name == env.features[pot.feature_idx], 'Potion\'s feature id does not match feature name.'
        num_samples = 1000
        num_tests = 10
        for i in range(env.num_features):
            num_successes = 0
            for _ in range(num_tests):
                samples = []
                for _ in range(num_samples):
                    samples.append(env.world.actions[i].distribution.rvs())
                _, p_value = stats.kstest(samples, distributions[i].cdf)
                if p_value > 0.05:
                    num_successes += 1
            assert num_successes > 8, f'With p-values over 0.05, the potion that affects "{env.features[i]}" failed the Kolmogorov-Smirnov test {num_tests - num_successes} times out of {num_tests}. Run the test again to see if this is a false positive.'