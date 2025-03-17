import numpy as np
import pytest
import scipy.stats as stats

from src.alchemy.classic_alchemy_env import ClassicAlchemyEnv
from src.alchemy.potion import Potion

REWARDS = {'shiny': 1, 'hard': 0.5}

class TestClassicAlchemyEnv():
    def test_generate_1(self):
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
        env = ClassicAlchemyEnv(['shiny', 'hard'])
        assert env.reward_func is None
        
        env.set_reward_func({'shiny': 1, 'hard': 0.5})
        assert env.reward_func is not None
        assert pytest.approx(env.reward_func(np.array([0, 0]))) == 0   
        assert pytest.approx(env.reward_func(np.array([1, 0]))) == 1   
        assert pytest.approx(env.reward_func(np.array([0, 1]))) == 0.5 
        assert pytest.approx(env.reward_func(np.array([1, 1]))) == 1.5 
    
    def test_reward_func_2(self):
        '''Test the case where the reward function is set with only one feature.
        The set reward function should automatically set the reward for the other
        features to 0.
        '''
        env = ClassicAlchemyEnv(['shiny', 'hard'])
        env.set_reward_func({'shiny': 1, 'hard': 0.5})
        assert env.reward_func is not None
        assert pytest.approx(env.reward_func(np.array([0, 0]))) == 0
        assert pytest.approx(env.reward_func(np.array([1, 0]))) == 1
    
    def test_reward_func_3(self):
        '''Test case where input state to reward function is a 
        different length from the number of features.
        '''
        env = ClassicAlchemyEnv()
        env.set_reward_func(REWARDS)
        with pytest.raises(AssertionError):
            env.reward_func(np.array([0, 0, 0, 0]))
        
        with pytest.raises(AssertionError):
            env.reward_func(np.array([0]))
            
    def test_reset_1(self):
        '''Test that the reset function adequately checks that 
        the input world is valid.
        '''
        env = ClassicAlchemyEnv()
        pot = Potion(0, 'shiny', stats.binom(1, 1))
        
        # wrong lengths of tuple elements
        with pytest.raises(AssertionError):
            env.reset(([], []))
            
        # number of actions does not match the action space
        with pytest.raises(AssertionError):
            env.reset(([pot] * (env.action_space.n + 2), []))
            
        # invalid action in blocked pair (out of range)
        with pytest.raises(AssertionError):
            env.reset((
                [pot] * env.action_space.n, 
                [(np.random.randint(0, 2, env.n), env.action_space.n)]
            ))
            
        # too many blocked pairs
        with pytest.raises(AssertionError):
            env.reset((
                [pot] * env.action_space.n, 
                [(np.random.randint(0, 2, env.n), 0)] * (env.max_blocks + 1)
            ))           
            
        # state shape does not match the number of features
        with pytest.raises(AssertionError):
            env.reset((
                [pot] * env.action_space.n, 
                [(np.random.randint(0, 2, env.n + 1), 0)] * (env.max_blocks + 1)
            ))  
            
        env.reset((
                [pot] * env.action_space.n, 
                [(np.random.randint(0, 2, env.n), 0)] * (env.max_blocks)
            ))   
        env.reset(([pot] * env.action_space.n, []))
        
    def test_step_1(self):
        '''Test the step function with valid non-ending actions. 
        '''
        env = ClassicAlchemyEnv()
        env.reset()
        env.set_reward_func(REWARDS)
        
        # do 50 steps without ending the episode
        total_reward = 0
        for i in range(50):
            # sample valid non-ending action
            action = env.action_space.sample()
            while action == env.action_space.n - 1:
                action = env.action_space.sample()
            state, reward, done = env.step(action)
            total_reward += reward
            assert not done, "The episode is done, but it should not be."
            assert env.observation_space.contains(state), "State is not in the observation space."
            assert total_reward == pytest.approx(env.time_cost * (i+1)), f"Reward is incorrect."
            assert action in env.stale_actions, "Action should be in env.stale_actions after step."
        
    def test_step_2(self):
        '''Test the step function with ending action.
        '''
        env = ClassicAlchemyEnv()
        env.reset()
        env.set_reward_func(REWARDS)
        action = env.action_space.n - 1
        state, reward, done = env.step(action)
        assert done, "The episode is not done, but it should be."
        assert env.observation_space.contains(state), "State is not in the observation space."
        assert reward <= 1.5 and reward >= 0, f"Reward out of range: {reward}"
        
    def test_sample_action_1(self):
        '''Test that the actions are valid, and that 
        non-stale actions are sampled when stale_ok is False.
        '''
        env = ClassicAlchemyEnv()
        env.set_reward_func(REWARDS)
        env.reset()
        for _ in range(50):
            action = env.sample_action(stale_ok=False)
            assert env.action_space.contains(action), "Action is not in the action space."
            assert action not in env.stale_actions, "Action is stale. It shouldn't be."
            
            _, _, is_finished = env.step(action)
            if is_finished:
                break