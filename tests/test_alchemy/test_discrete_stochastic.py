import numpy as np
from src.alchemy.discreate_stochastic_alchemy_env import DiscreteStochasticAlchemyEnv

class TestDiscreteStochasticAlchemyEnv():
    def test_generate_world(self):
        features = ['shiny', 'hard', 'hazy']
        probabilities = [0.1, 0.2, 0.8]
        env = DiscreteStochasticAlchemyEnv(features)
        env.reset(env.generate_from_probabilities(probabilities))
        assert len(env.world.actions) == len(features) * 2 + 1, 'The number of actions is incorrect.'
        assert len(env.world.blocked_pairs) <= env.max_blocks, 'Too many blocked pairs.'
        for pot in env.world.actions[:-1]:
            assert pot.feature_name == features[pot.feature_idx], 'Potion\'s feature id does not match feature name.'
        num_samples = 1000
        num_tests = 10
        for i in range(len(features)):
            pos_samples = []
            neg_samples = []
            idx = i * 2
            p = probabilities[i]
            q = 1 - p
            standard_error_of_mean = (p*q / num_samples)**0.5
            pos_successes = 0
            neg_successes = 0
            for _ in range(num_tests):
                pos_samples.extend(env.world.actions[idx].distribution.rvs(num_samples))
                neg_samples.extend(env.world.actions[idx + 1].distribution.rvs(num_samples))
                if np.mean(pos_samples) - p < 3*standard_error_of_mean:
                    pos_successes += 1
                if np.mean(neg_samples) - q < 3*standard_error_of_mean:
                    neg_successes += 1
            assert pos_successes > 8, f'The potion to add "{features[i]}" fell outside (3 * standard error of mean) {num_tests - pos_successes} times out of {num_tests}. Run the test again to see if this is a false positive.'
            assert neg_successes > 8, f'The potion to remove "{features[i]}" fell outside (3 * standard error of mean) {num_tests - neg_successes} times out of {num_tests}. Run the test again to see if this is a false positive.'