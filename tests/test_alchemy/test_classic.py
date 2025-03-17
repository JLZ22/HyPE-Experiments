import numpy as np
import pytest

from src.alchemy.classic_alchemy_env import ClassicAlchemyEnv

class TestClassicAlchemyEnv():
    def test_generate(self):
        '''Test the generation of the world in the classic alchemy environment. 
        Make sure that the sizes of the actions and blocked pairs are correct.
        '''
        env = ClassicAlchemyEnv()
        wrld = env.world
        assert wrld[0] is not None
        assert wrld[1] is not None
        assert len(wrld[0]) == env.action_space.n
        assert len(wrld[1]) <= env.max_blocks
        for act in wrld[0]:
            assert act is not None
        for pair in wrld[1]:
            assert pair is not None
            assert len(pair) == 2
            assert pair[0] is not None
            assert pair[1] is not None
            assert env.observation_space.contains(pair[0])
            assert env.action_space.contains(pair[1])
            
    def test_reward_func_1(self):
        '''Test the case where the reward function is set with all features.
        '''
        env = ClassicAlchemyEnv(['luster', 'hardness'])
        assert env.reward_func is None
        
        rewards = {'luster': 1, 'hardness': 0.5}
        env.set_reward_func(rewards)
        assert env.reward_func is not None
        assert env.reward_func(np.array([0, 0])) == 0
        assert env.reward_func(np.array([1, 0])) == 1
        assert env.reward_func(np.array([0, 1])) == 0.5
        assert env.reward_func(np.array([1, 1])) == 1.5
    
    def test_reward_func_2(self):
        '''Test the case where the reward function is set with only one feature.
        The set reward function should automatically set the reward for the other
        features to 0.
        '''
        env = ClassicAlchemyEnv(['luster', 'hardness'])
        rewards = {'luster': 1}
        env.set_reward_func(rewards)
        assert env.reward_func is not None
        assert env.reward_func(np.array([0, 0])) == 0
        assert env.reward_func(np.array([1, 0])) == 1
    
    def test_reward_func_3(self):
        '''Test case where input state to reward function is a 
        different length from the number of features.
        '''
        env = ClassicAlchemyEnv(['luster', 'hardness'])
        rewards = {'luster': 1, 'hardness': 0.5}
        env.set_reward_func(rewards)
        with pytest.raises(AssertionError):
            env.reward_func(np.array([0, 0, 0]))
        
        with pytest.raises(AssertionError):
            env.reward_func(np.array([0]))