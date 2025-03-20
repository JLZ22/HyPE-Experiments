import numpy as np

from src.train import AlchemyDataset

class TestAlchemyDataset():
    def test_add_transition(self):
        dataset = AlchemyDataset()
        num_points = 10
        num_features = 3
        for _ in range(num_points):
            observation = np.random.randint(0, 2, num_features)
            action = np.random.randint(0, 2, num_features)
            next_observation = np.random.randint(0, 2, num_features)
            reward = np.random.random()
            term = np.random.randint(0, 2)
            state = np.random.randint(0, 2, num_features)
            dataset.add_transition(observation, action, next_observation, reward, term, state)
        
        assert len(dataset) == num_points, f'Dataset should have {num_points} points.'
        for i in range(num_points):
            assert dataset[i]['observation'].shape == (num_features,), f'Observation shape is inaccurate.'
            assert dataset[i]['action'].shape == (num_features,), f'Action shape shape is inaccurate.'
            assert dataset[i]['next_observation'].shape == (num_features,), f'Next observation shape is inaccurate.'
            assert dataset[i]['reward'].shape == (1,), f'Reward shape is inaccurate.'
            assert dataset[i]['term'].shape == (1,), f'Term shape is inaccurate.'
            assert dataset[i]['state'].shape == (num_features,), f'State shape is inaccurate.'