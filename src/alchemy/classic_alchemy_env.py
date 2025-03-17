import gymnasium as gym 
import numpy as np
import scipy.stats as stats
from typing import Optional

from .custom_types import World
from .potion import Potion

class ClassicAlchemyEnv(gym.Env):
    def __init__(
        self,
        feature_labels=['shiny', 'hard', 'hazy'],
        time_cost=-0.05,
        max_blocks=3,
    ):
        '''Initialize a classic alchemy environment. The action space is 
        2 * len(feature_labels) + 1. One action is to add a feature to the 
        rock and the other is to remove a feature. The last action is to 
        submit the rock for evaluation. The observation space is a binary 
        list of length len(feature_labels) indicating which features are
        present in the rock. This environment is deterministic, so the 
        distribution that the potion uses is a binomial distribution with
        probabilities 1 and 0 for adding and removing a feature respectively.

        Args:
            feature_labels (list, optional): The features of the alchemy world. Defaults to ['luster', 'hardness', 'weight', 'shape', 'color', 'clarity'].
            time_cost (float, optional): The penalty to the reward for every time step. Defaults to -0.05.
            max_blocks (int, optional): The largest number of (state, action) pairs that
            can be blocked. Defaults to 5.
        '''
        self.features = feature_labels
        self.n = len(feature_labels)
        self.time_cost = time_cost
        self.max_blocks = max_blocks
        self.finished = False
        self.action_space = gym.spaces.Discrete(2 * self.n + 1)
        self.observation_space = gym.spaces.MultiBinary(self.n)
        self.stale_actions = set() # set of actions that have been already taken in this episode
        
        self.curr_state = None 
        self.reward_func = None
        self.world = self.generate_world()
        self.reset()
    
    def set_reward_func(self, rewards: dict[str, float]):
        '''Set the reward function for the environment. If None, then the
        default reward function will be used.

        Args:
            rewards (dict[str, float]): The rewards for each feature.
        '''
        assert isinstance(rewards, dict), "The rewards must be a dictionary."
        for feature in self.features:
            if feature not in rewards:
                rewards[feature] = 0
        def reward_func(state: np.ndarray) -> float:
            '''The reward function given the current state of the environment 
            based on the given rewards for each feature.

            Args:
                state (np.ndarray): The current state of the rock.

            Returns:
                float: The reward for the current state.
                
            Raises:
                AssertionError: If the length of the state does not match the number of features.
            '''
            assert len(state) == len(self.features), "The state must have the same number of features as the environment."
            return sum([rewards[feature] * state[i] for i, feature in enumerate(self.features)])
        self.reward_func = reward_func
        
    def generate_world(self) -> World:
        '''Generate a random world for the environment.

        Returns:
            World: The list of Potions and a list of blocked (state, action) pairs.
        '''
        actions = []
        blocked_pairs = []
        # generate the deterministic actions for the environment
        i = 0
        while i < 2 * self.n:
            actions.append(Potion(i // 2, self.features[i // 2], stats.binom(1, 1)))
            i += 1
            actions.append(Potion(i // 2, self.features[i // 2], stats.binom(1, 0)))
            i += 1
        actions.append(Potion(2 * self.n, 'submit', stats.binom(1, 1)))
            
        # generate random blocked pairs 
        num_pairs = np.random.randint(0, self.max_blocks)
        
        for _ in range(num_pairs):
            state = np.random.randint(0, 2, self.n)
            action = np.random.randint(0, 2 * self.n)
            blocked_pairs.append((state, action))
        
        return actions, blocked_pairs
    
    def sample_initial_state(self) -> np.ndarray:
        '''Generate the initial state of the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        '''
        return np.random.randint(0, 2, self.n)
    
    def reset(
        self,
        new_wrld: Optional[World] = None
    ) -> np.ndarray:
        '''Reset the environment to an initial state. If a new world 
        is given, then the environment will be reset with that world.
        Otherwise, the environment will be reset with the same world.

        Args:
            new_wrld (Optional[World], optional): The new world to reset the environment with. Defaults to None.

        Returns:
            np.ndarray: The initial state of the environment.
        '''
        if new_wrld is not None:
            self.world = new_wrld
            assert len(new_wrld) == 2, "The world must have two elements, a list of actions and a list of blocked pairs."
            assert all([isinstance(act, Potion) for act in new_wrld[0]]), "The actions must be instances of the Potion class."
            assert all([len(pair) == 2 for pair in new_wrld[1]]), "The blocked pairs must be tuples of length 2."
            assert all([pair[0].shape == (self.n,) for pair in new_wrld[1]]), "The states in the blocked pairs must have the same shape as the observation space."
            assert all([pair[1] < self.action_space.n for pair in new_wrld[1]]), "The actions in the blocked pairs must be less than the size of the action space."
            assert len(new_wrld[0]) == self.action_space.n, "The number of actions must match the size of the action space."
            assert len(new_wrld[1]) <= self.max_blocks, "The number of blocked pairs must be less than or equal to the maximum number of blocks."
        self.finished = False
        self.stale_actions = set()
        self.curr_state = self.sample_initial_state()
        return self.curr_state
    
    def get_obs(self) -> np.ndarray:
        '''Return the observed current state 
        of the environment.

        Returns:
            np.ndarray: The state of the env.
        '''
        return self.curr_state.copy()
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        '''Take the given action in the environment and 
        return the new state.

        Args:
            action (int): The action to take in the environment. This 
            is also the index of the potion to apply.

        Returns:
            tuple[np.ndarray, float, bool]: The new observation of the environment, the reward for the action, and whether the episode is finished.
        '''
        if self.reward_func is None:
            raise RuntimeError("The reward function must be set before taking actions.")
        if self.finished:
            raise RuntimeError("The episode has already finished. You must call reset to start a new episode.")
        if action < 0 or not self.action_space.contains(action):
            raise ValueError("The action is out of range.")
        
        self.stale_actions.add(action)
        
        if action == 2 * self.n: # check if the action is to submit the rock
            self.finished = True
            reward = self.reward_func(self.curr_state) # reward is gained at final evaluation
            return self.get_obs(), reward, self.finished
        
        # check if the action is blocked
        for blocked_state, blocked_action in self.world[1]:
            if (
                np.array_equal(self.curr_state, blocked_state) 
                and 
                action == blocked_action
            ):
                reward = self.time_cost
                return self.get_obs(), reward, self.finished
        
        # any other valid action
        potion = self.world[0][action]
        self.curr_state = potion.use_on(self.curr_state)
        reward = self.time_cost
        return self.get_obs(), reward, self.finished
    
    def sample_action(self, stale_ok: bool = False) -> int:
        '''Sample an action from the action space. If stale_ok is False, 
        then the action will be one that changes the state of the rock. 

        Args:
            stale_ok (bool, optional): If True, the sample space will include actions
            that have already been taken. Defaults to False.

        Returns:
            int: The sampled action.
        '''
        if stale_ok:
            return self.action_space.sample()
        else:
            action = self.action_space.sample()
            while action in self.stale_actions:
                action = self.action_space.sample()
            return action