import numpy as np
from src.alchemy.discreate_stochastic_alchemy_env import DiscreteStochasticAlchemyEnv

class TestClassicalAlchemyEnv():
    def test_generate_world(self):
        features = ['shiny', 'hard', 'hazy']
        probabilities = [0.1, 0.2, 0.8]
        env = DiscreteStochasticAlchemyEnv(features)
        env.reset(env.generate_from_probabilities(probabilities))
        assert len(env.world.actions) == len(features) * 2 + 1, 'The number of actions is incorrect.'
        assert len(env.world.blocked_pairs) <= env.max_blocks, 'Too many blocked pairs.'
        num_samples = 1000
        for i in range(len(features)):
            pos_samples = []
            neg_samples = []
            idx = i * 2
            for _ in range(num_samples):
                pos_samples.append(env.world.actions[idx].distribution.rvs())
                neg_samples.append(env.world.actions[idx + 1].distribution.rvs())
            p = probabilities[i]
            q = 1 - p
            standard_error_of_mean = (p*q / num_samples)**0.5
        
            # this should cover 99.7% of the samples
            assert abs(np.mean(pos_samples) - p) < 3*standard_error_of_mean, f'Sample mean of positive potion for feature {features[i]} is incorrect.'
            assert abs(np.mean(neg_samples) - q) < 3*standard_error_of_mean, f'Sample mean of negative potion for feature {features[i]} is incorrect.'